"""Public pump reads, current dashboard refresh and cached prediction endpoints."""

import asyncio
import time

from fastapi import APIRouter, Header, HTTPException, Query, Request
from datetime import datetime, timedelta, timezone
from typing import List
import numpy as np
import pandas as pd

# Import worker state
from backend.workers.prediction_worker import _in_memory_cache, _pump_states
from backend.engine.data_quality import add_average_features
from backend.api.dashboard_payload import affected_readings, compact_dashboard_payload, json_safe, public_engineering_profile
from backend.api.issue_catalog import FAULT_LABELS, build_issue_tiles
from backend.config import settings
from backend.api.auth import authorize


router = APIRouter(prefix="/api/pumps", tags=["Pumps"])
_dashboard_refresh_locks: dict[str, asyncio.Lock] = {}
_dashboard_last_refresh: dict[str, float] = {}


async def _refresh_dashboard_pump(pump_id: str) -> None:
    """Coalesce simultaneous refreshes and cap Influx fetches per pump."""
    from backend.workers.prediction_worker import worker_instance

    lock = _dashboard_refresh_locks.setdefault(pump_id, asyncio.Lock())
    async with lock:
        last_refresh = _dashboard_last_refresh.get(pump_id)
        if last_refresh is not None and time.monotonic() - last_refresh < settings.REFRESH_MIN_INTERVAL_SECONDS:
            return
        try:
            await worker_instance.refresh_pump(pump_id)
        finally:
            _dashboard_last_refresh[pump_id] = time.monotonic()

def _dashboard_label(agg_status: dict) -> str:
    """Map internal pump state and faults to a client-facing status label."""
    quality = agg_status.get("data_quality") or {}
    if quality.get("status") == "FAIL":
        return "Data Quality Error"
    state = str(agg_status.get("state") or "").upper()
    if state in {"NORMAL", "PROVISIONAL_NORMAL"}:
        return "Normal"
    if state == "BASELINE_PENDING":
        return "Baseline Pending"
    if state == "CONFIGURATION_REQUIRED":
        return "Engineering Profile Required"
    if state == "PENDING_APPROVAL":
        return "Waiting for Engineer Approval"
    if state == "OFF":
        return "Pump Off"
    if state == "MODEL_ERROR":
        return "Analysis Unavailable"
    if state == "DATA_ERROR":
        return "Data Unavailable"
    if state == "STALE_DATA":
        return "No recent data"
    if state == "DATA_QUALITY_ERROR":
        return "Data Quality Error"
    if state == "BASELINE_INCOMPLETE":
        return "Baseline Incomplete"
    if state == "WARMUP":
        return "Waiting for Stable Data"
    if state == "MODEL_SCORE_UNAVAILABLE":
        return "Waiting for Analysis"
    if state in {"ANOMALY", "PROVISIONAL_ANOMALY"}:
        faults = agg_status.get("detected_faults") or []
        primary_faults = [fault for fault in faults if fault.get("role") != "supporting"] or faults
        if len(primary_faults) > 1:
            return "Multiple Issues Detected"
        if len(primary_faults) == 1:
            code = str(primary_faults[0].get("code") or "").upper()
            return FAULT_LABELS.get(code, primary_faults[0].get("label") or code or "Candidate Fault")
        return "Unusual Pump Behaviour"
    return "System Error"


def _summary_message(agg_status: dict, label: str) -> str:
    """Choose the clearest operator summary for the current pump state."""
    state = str(agg_status.get("state") or "").upper()
    state_reason = agg_status.get("state_reason")
    if state == "WARMUP":
        if state_reason and "all phase currents" in str(state_reason).lower():
            return state_reason
        return "Pump recently started. Waiting for enough stable ON readings before anomaly detection."
    return state_reason or "; ".join(agg_status.get("engineering_messages") or []) or label


def _public_state(state: str | None) -> str:
    """Collapse provisional internal states into the public state model."""
    normalized = str(state or "").upper()
    if normalized == "PROVISIONAL_NORMAL":
        return "NORMAL"
    if normalized == "PROVISIONAL_ANOMALY":
        return "ANOMALY"
    return normalized


def _public_status_payload(agg_status: dict) -> dict:
    """Apply read-time staleness and remove unsafe guidance from stale data."""
    payload = dict(agg_status)
    payload["state"] = _public_state(payload.get("state"))
    payload["model_status"] = _public_state(payload.get("model_status"))
    if payload["state"] in {"NORMAL", "ANOMALY", "OFF", "STALE_DATA"}:
        timestamp = pd.to_datetime(payload.get("data_timestamp"), utc=True, errors="coerce")
        if not pd.isna(timestamp):
            age_seconds = max(0.0, (datetime.now(timezone.utc) - timestamp.to_pydatetime()).total_seconds())
            if age_seconds > settings.MAX_TELEMETRY_AGE_SECONDS or payload["state"] == "STALE_DATA":
                payload.update({
                    "state": "STALE_DATA", "health_score": None,
                    "health": {"score": None, "band": None, "confidence": "UNAVAILABLE", "reason": "STALE_DATA", "groups": {}, "contributors": []},
                    "state_reason": "No recent data. Live condition and guidance are unavailable.",
                    "detected_faults": [], "engineering_flags": [], "engineering_messages": [],
                    "baseline_deviations": [], "early_warnings": [],
                    "future_electrical_risk": None, "electrical_maintenance": None,
                    "forecast": {"state": "SKIPPED_STALE_DATA"},
                    "data_age_seconds": age_seconds, "last_data_timestamp": timestamp.isoformat(),
                })
    return payload


def _detailed_issues(agg_status: dict) -> list[dict]:
    """Normalize faults, deviations and warnings into one issue list."""
    issues = []
    for fault in agg_status.get("detected_faults") or []:
        code = str(fault.get("code") or "").upper()
        issues.append({
            "type": "named_fault",
            "code": code,
            "label": FAULT_LABELS.get(code, fault.get("label") or code),
            "evidence": fault.get("evidence"),
            "source": fault.get("detection_source"),
        })
    for deviation in agg_status.get("baseline_deviations") or []:
        issues.append({
            "type": "baseline_deviation",
            "code": deviation.get("code"),
            "sensor": deviation.get("sensor"),
            "phase": deviation.get("phase"),
            "actual": deviation.get("actual", deviation.get("value")),
            "baseline_median": deviation.get("baseline_median"),
            "expected_low": deviation.get("expected_low", deviation.get("expected_min")),
            "expected_high": deviation.get("expected_high", deviation.get("expected_max")),
            "direction": deviation.get("direction"),
            "severity": deviation.get("severity"),
            "evidence": deviation.get("evidence"),
        })
    for warning in agg_status.get("early_warnings") or []:
        issues.append({
            "type": "early_warning",
            "code": warning.get("code"),
            "sensor": warning.get("sensor"),
            "phase": warning.get("phase"),
            "actual": warning.get("current_value", warning.get("value")),
            "baseline_median": warning.get("baseline_median"),
            "expected_low": warning.get("expected_low"),
            "expected_high": warning.get("expected_high"),
            "direction": warning.get("direction"),
            "severity": warning.get("severity"),
            "evidence": warning.get("trend_evidence") or warning.get("evidence"),
        })
    return issues

# get_v2_status removed as it is now live

@router.get("", response_model=List[str])
async def get_all_pumps():
    """Fetch the current pump IDs from Influx on each call (page load/refresh)."""
    from backend.workers.prediction_worker import worker_instance
    return await worker_instance.discover_pumps()


@router.get("/{pump_id}/events")
async def get_pump_events(
    pump_id: str,
    status: str = Query("ALL", pattern="^(ALL|ACTIVE|RESOLVED)$"),
    event_type: str = Query("ALL", alias="type", pattern="^(ALL|FAULT|EARLY_WARNING|SYSTEM)$"),
    limit: int = Query(50, ge=1, le=200),
):
    """Read persisted issue history separately from the latest dashboard state."""
    from backend.workers.prediction_worker import worker_instance

    repository = worker_instance.issue_event_repository
    if repository is None:
        raise HTTPException(status_code=503, detail="Issue history is currently unavailable.")
    return await asyncio.to_thread(repository.list_events, pump_id, status=status, event_type=event_type, limit=limit)

@router.get("/{pump_id}/prediction")
async def get_pump_prediction(
    pump_id: str,
    request: Request,
    refresh: bool = Query(False),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """Serve cached current/forecast state; only keyed refresh requests fetch again."""
    from backend.workers.prediction_worker import worker_instance
    if refresh and (x_api_key or settings.AUTH_DISABLED_FOR_DEV):
        authorize(request, x_api_key, "engineer")
    elif refresh:
        refresh = False
    if refresh:
        await worker_instance.refresh_pump(pump_id)
    elif pump_id not in _in_memory_cache:
        await worker_instance.initialize_pump(pump_id)
    if pump_id not in _in_memory_cache:
        raise HTTPException(status_code=404, detail=f"No recent data found for pump {pump_id} in cache.")
    agg_status = _public_status_payload(_in_memory_cache[pump_id])
    
    raw_status = str(agg_status.get("pump_status", "ON")).upper()
    state = agg_status.get("state", "NORMAL")
    if raw_status in ["OFF", "OFFLINE"]:
        raw_status = "OFF"
            
    health_score = agg_status.get("health_score", 100.0)
    
    raw_reason = "; ".join(f["label"] for f in agg_status.get("detected_faults", []))
    if state in ("ANOMALY", "PROVISIONAL_ANOMALY") and not raw_reason:
        raw_reason = "; ".join(d["evidence"] for d in agg_status.get("baseline_deviations", []))
    if state in ("ANOMALY", "PROVISIONAL_ANOMALY") and not raw_reason:
        raw_reason = (agg_status.get("model_evidence") or {}).get("evidence", "")
    if state in ("DATA_ERROR", "DATA_QUALITY_ERROR", "STALE_DATA", "MODEL_ERROR", "BASELINE_PENDING", "MODEL_SCORE_UNAVAILABLE"):
        raw_reason = "; ".join(agg_status.get("engineering_messages", []))

    client_response = {
        "pump_id": agg_status.get("pump_id"),
        "pump_status": raw_status,
        "running_status": raw_status,
        "state": state,
        "dashboard_label": _dashboard_label(agg_status),
        "summary": {
            "condition": _dashboard_label(agg_status),
            "message": _summary_message(agg_status, _dashboard_label(agg_status)),
            "is_anomalous": state == "ANOMALY",
        },
        "health_score": health_score,
        "health": agg_status.get("health"),
        "anomaly_detection": {
            "status": state,
            "root_cause": raw_reason
        },
        "engineering_flags": agg_status.get("engineering_flags", []),
        "engineering_messages": agg_status.get("engineering_messages", []),
        "detected_faults": agg_status.get("detected_faults", []),
        "baseline_deviations": agg_status.get("baseline_deviations", []),
        "early_warnings": agg_status.get("early_warnings", []),
        "baseline_statistics": agg_status.get("baseline_statistics"),
        "baseline_period": agg_status.get("baseline_period"),
        "baseline_pending_detail": agg_status.get("baseline_pending_detail"),
        "baseline_source": agg_status.get("baseline_source"),
        "baseline_quality": agg_status.get("baseline_quality"),
        "baseline_row_count": agg_status.get("baseline_row_count"),
        "baseline_target_row_count": agg_status.get("baseline_target_row_count"),
        "baseline_minimum_row_count": agg_status.get("baseline_minimum_row_count"),
        "baseline_session_count": agg_status.get("baseline_session_count"),
        "comparison_window_rows": agg_status.get("comparison_window_rows"),
        "model_status": agg_status.get("model_status"),
        "model_evidence": agg_status.get("model_evidence"),
        "anomaly_score": agg_status.get("anomaly_score"),
        "smoothed_score": agg_status.get("smoothed_score"),
        "score_threshold": agg_status.get("threshold"),
        "current_streak": agg_status.get("current_streak"),
        "informational_status": agg_status.get("informational_status", {}),
        "current_values": agg_status.get("current_values", {}),
        "baseline_ranges": agg_status.get("baseline_ranges", {}),
        "data_quality": agg_status.get("data_quality", {}),
        "data_quality_warnings": agg_status.get("data_quality_warnings", []),
        "data_age_seconds": agg_status.get("data_age_seconds"),
        "last_data_timestamp": agg_status.get("last_data_timestamp"),
        "live_context_diagnostics": agg_status.get("live_context_diagnostics", {}),
        "baseline_issues": agg_status.get("baseline_issues", []),
        "state_reason": agg_status.get("state_reason"),
        "alert_start_time": agg_status.get("alert_start_time"),
        "data_timestamp": agg_status.get("data_timestamp"),
        "last_timestamp": agg_status.get("data_timestamp"),
        "analysis_context": agg_status.get("analysis_context"),
        "last_on_timestamp": agg_status.get("last_on_timestamp"),
        "last_run": agg_status.get("last_run"),
        "status_timestamp": agg_status.get("status_timestamp"),
        "last_processed_timestamp": agg_status.get("last_processed_timestamp"),
        "baseline_approved": agg_status.get("baseline_approved", False),
        "engineering_profile": public_engineering_profile(agg_status.get("engineering_profile")),
        "forecast": agg_status.get("forecast"),
        "future_electrical_risk": agg_status.get("future_electrical_risk"),
        "electrical_maintenance": agg_status.get("electrical_maintenance"),
    }
    client_response["issue_tiles"] = build_issue_tiles(client_response)
    return json_safe(client_response)

async def build_full_dashboard_payload(
    pump_id: str,
    start: str = "-1h",
    refresh: bool = False,
    include_trends: bool = True,
) -> dict:
    """Assemble live readings and cached guidance without running Timestral here."""
    from backend.workers.prediction_worker import worker_instance
    if refresh:
        await _refresh_dashboard_pump(pump_id)
    elif pump_id not in _in_memory_cache:
        await worker_instance.initialize_pump(pump_id)
    if pump_id not in _in_memory_cache:
        raise HTTPException(status_code=404, detail=f"No cache found for pump {pump_id}.")

    agg_status = _public_status_payload(_in_memory_cache[pump_id])
    raw_status = str(agg_status.get("pump_status", "UNKNOWN")).upper()
    state = str(agg_status.get("state", "UNKNOWN")).upper()
    df = add_average_features(_pump_states.get(pump_id, pd.DataFrame()).copy())
    
    trim_hours = 1
    if start.startswith("-") and start.endswith("h"):
        try:
            trim_hours = int(start[1:-1])
        except ValueError:
            pass
            
    if not df.empty:
        df["_time"] = pd.to_datetime(df["_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["_time"]).sort_values("_time")
        anchor_time = df["_time"].max()
        cutoff = anchor_time.to_pydatetime() - timedelta(hours=trim_hours)
        df = df[(df["_time"] >= cutoff) & (df["_time"] <= anchor_time)]
    else:
        df = pd.DataFrame()

    def series(col_name):
        """Serialize one telemetry trend column with missing values preserved."""
        if df.empty or col_name not in df.columns:
            return []
        return [None if pd.isna(v) or not np.isfinite(v) else round(float(v), 4) for v in pd.to_numeric(df[col_name], errors="coerce")]

    timestamps = ([] if df.empty or "_time" not in df.columns
                  else [t.isoformat() for t in df["_time"]])
    trend_series = {
        "timestamps": timestamps,
        "current_r": series("current_r"),
        "current_y": series("current_y"),
        "current_b": series("current_b"),
        "current_avg": series("current_avg"),
        "voltage_r": series("voltage_r"),
        "voltage_y": series("voltage_y"),
        "voltage_b": series("voltage_b"),
        "voltage_avg": series("voltage_avg"),
        "pf_r": series("pf_r"),
        "pf_y": series("pf_y"),
        "pf_b": series("pf_b"),
        "pf_avg": series("pf_avg"),
        "total_kw": series("total_kw"),
        "flowrate": series("flowrate"),
    }

    label = _dashboard_label(agg_status)
    details = _detailed_issues(agg_status)
    current_values = agg_status.get("current_values", {}) or {}
    current_avg = (((current_values.get("current") or {}).get("avg"))
                   if isinstance(current_values, dict) else None)
    power = (((current_values.get("power") or {}).get("total_kw"))
             if isinstance(current_values, dict) else None)
    physical_running = bool(
        pd.to_numeric(current_avg, errors="coerce") > 0.5 and
        pd.to_numeric(power, errors="coerce") > 0
    )
    reported_status = raw_status
    analysis_context = agg_status.get("analysis_context") or "LIVE_ON_SESSION"
    if analysis_context == "LAST_ON_SESSION" and reported_status in {"OFF", "OFFLINE"}:
        physical_running = False
    status_tag_mismatch = (
        analysis_context != "LAST_ON_SESSION" and (
            (reported_status in {"OFF", "OFFLINE"} and physical_running) or
            (reported_status in {"ON", "RUNNING"} and not physical_running and state != "OFF")
        )
    )
    payload = {
        "pump_id": pump_id,
        "last_updated": agg_status.get("data_timestamp") or agg_status.get("last_processed_timestamp"),
        "running_status": "OFF" if raw_status in {"OFF", "OFFLINE"} else raw_status,
        "reported_pump_status": reported_status,
        "physical_running": physical_running,
        "status_tag_mismatch": status_tag_mismatch,
        "analysis_context": analysis_context,
        "last_on_timestamp": agg_status.get("last_on_timestamp"),
        "last_run": agg_status.get("last_run"),
        "status_timestamp": agg_status.get("status_timestamp"),
        "state": state,
        "dashboard_label": label,
        "health_score": agg_status.get("health_score"),
        "health": agg_status.get("health"),
        "data_age_seconds": agg_status.get("data_age_seconds"),
        "last_data_timestamp": agg_status.get("last_data_timestamp"),
        "summary": {
            "condition": label,
            "message": _summary_message(agg_status, label),
            "is_anomalous": state in {"ANOMALY", "PROVISIONAL_ANOMALY"},
        },
        "data_quality": agg_status.get("data_quality", {}),
        "data_quality_warnings": agg_status.get("data_quality_warnings", []),
        "live_context_diagnostics": agg_status.get("live_context_diagnostics", {}),
        "baseline_issues": agg_status.get("baseline_issues", []),
        "state_reason": agg_status.get("state_reason"),
        "current_values": agg_status.get("current_values", {}),
        "baseline_ranges": agg_status.get("baseline_ranges", {}),
        "detected_faults": agg_status.get("detected_faults", []),
        "baseline_deviations": agg_status.get("baseline_deviations", []),
        "early_warnings": agg_status.get("early_warnings", []),
        "baseline_pending_detail": agg_status.get("baseline_pending_detail"),
        "baseline_source": agg_status.get("baseline_source"),
        "baseline_quality": agg_status.get("baseline_quality"),
        "baseline_row_count": agg_status.get("baseline_row_count"),
        "baseline_target_row_count": agg_status.get("baseline_target_row_count"),
        "baseline_minimum_row_count": agg_status.get("baseline_minimum_row_count"),
        "detailed_issues": details,
        "trend_series": trend_series,
        "model_diagnostics": {
            "model_status": agg_status.get("model_status"),
            "anomaly_score": agg_status.get("anomaly_score"),
            "smoothed_score": agg_status.get("smoothed_score"),
            "score_threshold": agg_status.get("threshold"),
            "current_streak": agg_status.get("current_streak"),
            "comparison_window_rows": agg_status.get("comparison_window_rows"),
            "model_evidence": agg_status.get("model_evidence"),
        },
        "baseline_info": {
            "baseline_period": agg_status.get("baseline_period"),
            "baseline_statistics": agg_status.get("baseline_statistics"),
            "baseline_source": agg_status.get("baseline_source"),
            "baseline_quality": agg_status.get("baseline_quality"),
            "baseline_row_count": agg_status.get("baseline_row_count"),
            "baseline_target_row_count": agg_status.get("baseline_target_row_count"),
            "baseline_minimum_row_count": agg_status.get("baseline_minimum_row_count"),
            "baseline_session_count": agg_status.get("baseline_session_count"),
            "baseline_approved": agg_status.get("baseline_approved", False),
            "baseline_pending_detail": agg_status.get("baseline_pending_detail"),
            "baseline_issues": agg_status.get("baseline_issues", []),
        },
        "short_cycling": agg_status.get("short_cycling") or {},
        "engineering_profile": public_engineering_profile(agg_status.get("engineering_profile")),
        "forecast": agg_status.get("forecast") or {},
        "future_electrical_risk": agg_status.get("future_electrical_risk"),
        "electrical_maintenance": agg_status.get("electrical_maintenance"),
    }
    payload["issue_tiles"] = build_issue_tiles(payload)
    payload["affected_readings"] = affected_readings(payload, state)
    if not include_trends:
        payload.pop("trend_series", None)
    return payload


@router.get("/{pump_id}/dashboard")
async def get_pump_dashboard(
    pump_id: str,
    start: str = "-1h",
    refresh: bool = Query(False),
    include_trends: bool = True,
):
    """Return the compact operator contract; diagnostics live under /api/admin."""
    payload = await build_full_dashboard_payload(
        pump_id,
        start=start,
        refresh=refresh,
        include_trends=include_trends,
    )
    return json_safe(compact_dashboard_payload(payload, include_trends=include_trends))
