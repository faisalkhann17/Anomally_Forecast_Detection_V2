"""APIs for pump fleet inspection and detailed diagnostics."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.auth import require_admin
from backend.api.routes.pumps import _dashboard_label, _public_status_payload, build_full_dashboard_payload
from backend.workers.prediction_worker import (
    _in_memory_cache,
    _operating_event_states,
    _pump_states,
    worker_instance,
)


logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/admin/pumps",
    tags=["Super Admin Diagnostics"],
    dependencies=[Depends(require_admin)],
)


def _pump_summary(pump_id: str) -> dict:
    """Build a cache-only fleet summary without triggering telemetry reads."""
    cached = _in_memory_cache.get(pump_id)
    if not cached:
        return {
        "pump_id": pump_id,
            "state": "NOT_LOADED",
            "dashboard_label": "Not Loaded",
            "running_status": "UNKNOWN",
            "last_updated": None,
            "fault_count": 0,
            "warning_count": 0,
            "baseline_quality": None,
            "data_quality_status": "UNKNOWN",
        }
    status = _public_status_payload(cached)
    return {
        "pump_id": pump_id,
        "state": status.get("state") or "UNKNOWN",
        "dashboard_label": _dashboard_label(status),
        "running_status": status.get("pump_status") or "UNKNOWN",
        "last_updated": status.get("data_timestamp") or status.get("last_processed_timestamp"),
        "fault_count": len(status.get("detected_faults") or []),
        "warning_count": len(status.get("early_warnings") or []),
        "baseline_quality": status.get("baseline_quality"),
        "data_quality_status": (status.get("data_quality") or {}).get("status") or "UNKNOWN",
    }


def _reason_code(payload: dict) -> str:
    """Convert internal pipeline state into an actionable diagnostic code."""
    state = str(payload.get("state") or "UNKNOWN").upper()
    pending = payload.get("baseline_pending_detail") or {}
    if state in {"CONFIGURATION_REQUIRED", "PENDING_APPROVAL", "STALE_DATA", "OFF", "DATA_QUALITY_ERROR", "DATA_ERROR"}:
        return state
    if state in {"WARMUP", "MODEL_SCORE_UNAVAILABLE"}:
        return "INSUFFICIENT_LIVE_SESSION_ROWS"
    if state in {"BASELINE_PENDING", "BASELINE_INCOMPLETE"}:
        if pending.get("error"):
            return "BASELINE_ENROLLMENT_FAILED"
        if not pending.get("fetched_history_rows"):
            return "NO_HISTORICAL_DATA"
        if pending.get("rejected_rows", 0) > pending.get("valid_running_rows", 0):
            return "TOO_MANY_INVALID_ROWS"
        return "INSUFFICIENT_VALID_ON_ROWS"
    return "READY" if state in {"NORMAL", "ANOMALY"} else state


def _recommended_actions(code: str, pending: dict) -> list[str]:
    """Return operator actions appropriate for a diagnostic reason code."""
    actions = {
        "CONFIGURATION_REQUIRED": "Create an engineering profile for this pump.",
        "PENDING_APPROVAL": "Ask an authorized engineer to approve the profile.",
        "STALE_DATA": "Check telemetry delivery and refresh after a new reading arrives.",
        "OFF": "No live action required; analysis resumes when the pump runs.",
        "DATA_QUALITY_ERROR": "Correct invalid or missing telemetry fields.",
        "DATA_ERROR": "Check InfluxDB connectivity and the selected pump ID.",
        "INSUFFICIENT_LIVE_SESSION_ROWS": "Collect more valid readings in the current ON session.",
        "NO_HISTORICAL_DATA": "Confirm that the pump is writing ON telemetry to InfluxDB.",
        "TOO_MANY_INVALID_ROWS": "Review rejected sensor fields and correct their source.",
        "BASELINE_ENROLLMENT_FAILED": "Review enrollment logs and baseline storage.",
        "READY": "No pipeline action required.",
    }
    if code == "INSUFFICIENT_VALID_ON_ROWS":
        return [f"Collect {pending.get('rows_needed_for_minimum', 'more')} additional valid ON readings."]
    return [actions.get(code, "Review pipeline diagnostics and backend logs.")]


@router.get("")
async def get_admin_pumps():
    """Fetch pump IDs from Influx, then return cache-only per-pump summaries."""
    pump_ids = await worker_instance.discover_pumps()
    return [_pump_summary(pump_id) for pump_id in pump_ids]


@router.get("/{pump_id}/diagnostics")
async def get_pump_diagnostics(
    pump_id: str,
    start: str = "-1h",
    refresh: bool = Query(False),
    include_trends: bool = Query(True),
):
    """Return full authenticated diagnostics for one pump pipeline."""
    payload = await build_full_dashboard_payload(
        pump_id,
        start=start,
        refresh=refresh,
        include_trends=include_trends,
    )
    pending = payload.get("baseline_pending_detail") or {}
    reason_code = _reason_code(payload)
    frame = _pump_states.get(pump_id)
    event_frame = _operating_event_states.get(pump_id)
    profile = payload.get("engineering_profile") or {}
    payload["diagnostics"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "state": payload.get("state"),
            "reason": payload.get("state_reason"),
            "reason_code": reason_code,
            "recommended_actions": _recommended_actions(reason_code, pending),
            "profile_approval": profile.get("approval_status") or "NOT_CONFIGURED",
        },
        "pipeline_stages": {
            "latest_data_quality": (payload.get("data_quality") or {}).get("status") or "UNKNOWN",
            "forecast": (payload.get("forecast") or {}).get("state") or "UNKNOWN",
            "maintenance": (payload.get("electrical_maintenance") or {}).get("state") or "UNKNOWN",
        },
        "row_counts": {
            "live_rows_returned": 0 if frame is None else len(frame),
            "short_cycling_rows_available": 0 if event_frame is None else len(event_frame),
            "history_rows_fetched": pending.get("fetched_history_rows"),
            "valid_running_rows": pending.get("valid_running_rows"),
            "rejected_history_rows": pending.get("rejected_rows"),
        },
        "rejected_history": pending.get("rejection_breakdown") or {},
        "baseline_progress": {
            "minimum_progress_pct": pending.get("minimum_progress_pct"),
            "target_progress_pct": pending.get("target_progress_pct"),
            "rows_needed_for_minimum": pending.get("rows_needed_for_minimum"),
            "rows_needed_for_target": pending.get("rows_needed_for_target"),
        },
        "lookback": worker_instance.influx_service.get_lookback_diagnostics(pump_id),
        "issue_event_store_ready": worker_instance.issue_event_repository is not None,
    }
    return payload


@router.post("/{pump_id}/reinitialize")
async def reinitialize_pump(pump_id: str, force_baseline: bool = Query(False)):
    """Clear runtime state and initialize one pump; optionally rebuild its baseline."""
    _pump_states.pop(pump_id, None)
    _in_memory_cache.pop(pump_id, None)
    _operating_event_states.pop(pump_id, None)

    try:
        await worker_instance.initialize_pump(pump_id, force_enroll=force_baseline)
    except Exception as exc:
        logger.exception("Pump reinitialization failed for %s", pump_id)
        raise HTTPException(
            status_code=500,
            detail="Pump reinitialization failed.",
        ) from exc
    if pump_id not in _in_memory_cache:
        raise HTTPException(status_code=404, detail=f"Unknown pump {pump_id}: no profile and no telemetry.")
    return {
        "status": "REFRESHED",
        "pump_id": pump_id,
        "baseline_unchanged": not force_baseline,
    }

