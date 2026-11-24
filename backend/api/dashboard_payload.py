"""Operator-facing projection of the full dashboard response."""
import math

from backend.api.issue_catalog import build_issue_tiles


def _operator_current_values(values: dict | None) -> dict:
    """Return only sensor values rendered by operator KPIs and telemetry table."""
    values = values or {}
    result = {}
    for group in ("current", "voltage", "pf"):
        source = values.get(group) or {}
        result[group] = {
            key: source.get(key)
            for key in ("r", "y", "b", "avg", "available_phases")
            if key in source
        }
    result["power"] = {"total_kw": (values.get("power") or {}).get("total_kw")}
    result["flow"] = {"flowrate": (values.get("flow") or {}).get("flowrate")}
    return result


def _operator_health(health: dict | None) -> dict:
    """Expose only health fields intended for the operator dashboard."""
    source = health or {}
    return {
        "score": source.get("score"),
        "band": source.get("band"),
        "confidence": source.get("confidence", "UNAVAILABLE"),
        "contributors": source.get("contributors") or [],
        "model_flagged": bool(source.get("model_flagged")),
    }


def _metric_for_reading(item: dict) -> str | None:
    """Map an issue code to the KPI measurement it affects."""
    code = str(item.get("code") or item.get("field") or item.get("sensor") or "").upper()
    if code.startswith("CURRENT_") or code in {
        "CURRENT", "PHASE_CURRENT_IMBALANCE", "PHASE_LOSS", "MOTOR_OVERLOAD",
        "SHORT_CYCLING",
    }:
        return "current"
    if code.startswith("VOLTAGE_") or code in {
        "VOLTAGE", "VOLTAGE_IMBALANCE", "VOLTAGE_PHASE_LOSS",
        "UNDER_VOLTAGE", "OVER_VOLTAGE",
    }:
        return "voltage"
    if code.startswith("PF_") or code in {"PF", "LOW_PF"}:
        return "pf"
    if code.startswith(("TOTAL_KW", "POWER_")) or code in {"POWER", "POWER_KW"}:
        return "power"
    if code.startswith(("FLOWRATE", "FLOW_")) or code in {
        "FLOW", "DRY_RUN", "FLOW_RESTRICTION"
    }:
        return "flow"
    return None


def affected_readings(payload: dict, state: str) -> list[dict]:
    """Build the highlighted KPI list from faults and early warnings."""
    level = "ALERT" if state == "ANOMALY" else "WARNING"
    source = (payload.get("baseline_deviations") or []) if state == "ANOMALY" else (
        payload.get("early_warnings") or [] if state == "NORMAL" else []
    )
    grouped = {}
    for item in source:
        metric = _metric_for_reading(item)
        if metric is None:
            continue
        candidate = {
            "metric": metric,
            "direction": str(item.get("direction") or "OUTSIDE_RANGE").upper(),
            "level": level,
        }
        severity = float(item.get("severity") or 0)
        previous = grouped.get(metric)
        if previous is None or severity > previous[0]:
            grouped[metric] = (severity, candidate)

    if state == "ANOMALY":
        fault_metrics = {
            "DRY_RUN": ("flow", "current", "power"),
            "LOW_PF": ("pf",),
            "PHASE_CURRENT_IMBALANCE": ("current",),
            "VOLTAGE_IMBALANCE": ("voltage",),
            "PHASE_LOSS": ("current",),
            "VOLTAGE_PHASE_LOSS": ("voltage",),
            "UNDER_VOLTAGE": ("voltage",),
            "OVER_VOLTAGE": ("voltage",),
            "MOTOR_OVERLOAD": ("current", "power"),
            "FLOW_RESTRICTION": ("flow", "current", "power"),
            "SHORT_CYCLING": ("current",),
        }
        for fault in payload.get("detected_faults") or []:
            code = str(fault.get("code") or "").upper()
            for metric in fault_metrics.get(code, ()):
                if metric not in grouped:
                    grouped[metric] = (0, {
                        "metric": metric,
                        "direction": "OUTSIDE_RANGE",
                        "level": "ALERT",
                    })
    return [item for _, item in grouped.values()]


PUBLIC_PROFILE_FIELDS = (
    "pump_id", "version", "updated_at", "approval_status", "approved_by", "approved_at",
    "rated_power_kw", "rated_voltage_v", "rated_current_a", "rated_frequency_hz",
    "phase_count", "voltage_type", "operating_limits",
)


def public_engineering_profile(profile: dict | None) -> dict | None:
    """Whitelist profile fields safe for unauthenticated dashboard output."""
    if not isinstance(profile, dict):
        return None
    return {field: profile[field] for field in PUBLIC_PROFILE_FIELDS if field in profile}


def compact_dashboard_payload(payload: dict, *, include_trends: bool = True) -> dict:
    """Project the internal worker payload into the public dashboard schema."""
    state = str(payload.get("state") or "UNKNOWN").upper()
    result = {
        "pump_id": payload.get("pump_id"),
        "last_updated": payload.get("last_updated"),
        "running_status": payload.get("running_status"),
        "state": state,
        "health_score": payload.get("health_score"),
        "health": _operator_health(payload.get("health")),
        "data_age_seconds": payload.get("data_age_seconds"),
        "last_data_timestamp": payload.get("last_data_timestamp"),
        "data_quality_warnings": payload.get("data_quality_warnings") or [],
        "dashboard_label": payload.get("dashboard_label"),
        "analysis_context": payload.get("analysis_context"),
        "last_on_timestamp": payload.get("last_on_timestamp"),
        # While the pump is off: readings and health from its last run.
        "last_run": None if not payload.get("last_run") else {
            "ended_at": payload["last_run"].get("ended_at"),
            "current_values": _operator_current_values(payload["last_run"].get("current_values")),
            "health": _operator_health(payload["last_run"].get("health")),
        },
        "status_timestamp": payload.get("status_timestamp"),
        "current_values": _operator_current_values(payload.get("current_values")),
        "baseline_ranges": payload.get("baseline_ranges") or {},
        "issue_tiles": build_issue_tiles(payload),
    }
    faults = payload.get("detected_faults") or []
    if faults:
        result["detected_faults"] = faults
    warnings = payload.get("early_warnings") or []
    if state == "NORMAL" and warnings:
        result["early_warnings"] = warnings
    affected = affected_readings(payload, state)
    if affected:
        result["affected_readings"] = affected
    pending = payload.get("baseline_pending_detail")
    if state in {"BASELINE_PENDING", "BASELINE_INCOMPLETE"} and pending:
        result["baseline_pending_detail"] = pending
    quality = payload.get("data_quality") or {}
    if state == "DATA_QUALITY_ERROR" or str(quality.get("status") or "").upper() == "FAIL":
        result["data_quality"] = {
            key: quality[key]
            for key in ("status", "issues", "invalid_fields", "missing_fields",
                        "unavailable_fields", "hyphen_fields")
            if key in quality
        }
    profile = payload.get("engineering_profile")
    if profile:
        result["engineering_profile"] = public_engineering_profile(profile)
    forecast = payload.get("forecast") or {}
    selected_forecast = forecast.get("long_term") if isinstance(forecast.get("long_term"), dict) else forecast
    if selected_forecast:
        result["forecast"] = {
            key: selected_forecast.get(key)
            for key in ("state", "history_points", "calendar_points", "on_history_points", "history_duration_hours",
                        "interval_minutes", "horizon_minutes", "conditional_on_pump_on", "detail", "generated_at")
            if key in selected_forecast
        }
        if "generated_at" not in result["forecast"] and forecast.get("generated_at"):
            result["forecast"]["generated_at"] = forecast["generated_at"]
    risk = payload.get("future_electrical_risk")
    if risk:
        result["future_electrical_risk"] = risk
    maintenance = payload.get("electrical_maintenance")
    if maintenance:
        result["electrical_maintenance"] = maintenance
    if include_trends:
        result["trend_series"] = payload.get("trend_series") or {}
    return result


def json_safe(value):
    """Replace NaN/+-inf anywhere in a response with None so JSON stays valid."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        import numpy as np
        if isinstance(value, np.floating) and not np.isfinite(value):
            return None
    except ImportError:  # pragma: no cover
        pass
    return value
