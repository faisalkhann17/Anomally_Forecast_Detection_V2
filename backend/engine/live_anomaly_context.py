"""Build spike-resistant live context without replacing engineer limits."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from backend.engine.data_quality import SENSOR_FIELDS, add_average_features, data_quality_firewall
from backend.engine.electrical_profile import enriched_frame
from backend.engine.readable_text import count, field_name, with_unit


LIVE_CONTEXT_ROWS = 350
SESSION_LOOKBACK_MINUTES = 10
SESSION_MAX_ROWS = 15
SESSION_MIN_ROWS = 5
SESSION_GAP_MINUTES = 30

ROBUST_SCALE_FLOORS = {
    "current_r": 0.10, "current_y": 0.10, "current_b": 0.10,
    "voltage_r": 2.0, "voltage_y": 2.0, "voltage_b": 2.0,
    "pf_r": 0.01, "pf_y": 0.01, "pf_b": 0.01,
    "total_kw": 0.05, "flowrate": 0.20,
}


def _usable_on_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return clean, physically-running ON rows for live context calculations."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    work = enriched_frame(frame)
    work["_time"] = pd.to_datetime(work.get("_time"), utc=True, errors="coerce")
    work = work.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last")
    status_on = work.get("pump_status", pd.Series("", index=work.index)).astype(str).str.upper().str.strip().isin(["ON", "RUNNING"])
    currents = work[["current_r", "current_y", "current_b"]].apply(pd.to_numeric, errors="coerce")
    power = pd.to_numeric(work.get("total_kw"), errors="coerce")
    physical_on = currents.mean(axis=1).gt(0.5) | power.gt(0.1)
    quality_ok = work.apply(lambda row: data_quality_firewall(row)["status"] != "FAIL", axis=1)
    return work.loc[status_on & physical_on & quality_ok].copy()


def current_session_summary(
    frame: pd.DataFrame,
    *,
    minutes: int = SESSION_LOOKBACK_MINUTES,
    max_rows: int = SESSION_MAX_ROWS,
    min_rows: int = SESSION_MIN_ROWS,
) -> tuple[pd.Series | None, int]:
    """Median the latest stable ON-session rows so one spike cannot dominate."""
    work = _usable_on_rows(frame)
    if work.empty:
        return None, 0
    gaps = work["_time"].diff().gt(timedelta(minutes=SESSION_GAP_MINUTES))
    session_ids = gaps.cumsum()
    current = work.loc[session_ids.eq(session_ids.iloc[-1])]
    cutoff = current["_time"].iloc[-1] - timedelta(minutes=int(minutes))
    current = current.loc[current["_time"].ge(cutoff)].tail(int(max_rows))
    if len(current) < min_rows:
        return None, len(current)
    fields = [field for field in SENSOR_FIELDS if field in current.columns]
    summary = current[fields].apply(pd.to_numeric, errors="coerce").median()
    summary["_time"] = current["_time"].iloc[-1]
    summary["pump_status"] = "ON"
    # Health scoring reads current_avg/voltage_avg/pf_avg, so derive them from
    # the median phases exactly as the latest-reading path does.
    return add_average_features(enriched_frame(pd.DataFrame([summary]))).iloc[0], len(current)


def _approved_ranges(profile: dict) -> dict[str, tuple[float, float]]:
    """Engineer-approved range per phase field; values outside are handled by fault rules."""
    limits = profile["operating_limits"]
    return {
        "current_r": (limits["current_min_a"], limits["current_max_a"]),
        "current_y": (limits["current_min_a"], limits["current_max_a"]),
        "current_b": (limits["current_min_a"], limits["current_max_a"]),
        "voltage_r": (limits["voltage_min_v"], limits["voltage_max_v"]),
        "voltage_y": (limits["voltage_min_v"], limits["voltage_max_v"]),
        "voltage_b": (limits["voltage_min_v"], limits["voltage_max_v"]),
        "pf_r": (limits["power_factor_min"], 1.0),
        "pf_y": (limits["power_factor_min"], 1.0),
        "pf_b": (limits["power_factor_min"], 1.0),
        "total_kw": (limits["total_kw_min"], limits["total_kw_max"]),
    }


def robust_shift_warnings(
    frame: pd.DataFrame,
    reading: pd.Series | None,
    profile: dict,
    *,
    minimum_z: float = 4.0,
) -> list[dict]:
    """Describe sudden shifts within the current run that remain inside approved limits.

    The reference is the current ON session only (a stop longer than
    SESSION_GAP_MINUTES starts a new one), so normal day-to-day load changes
    after a restart are not reported as sudden shifts.
    """
    if reading is None:
        return []
    work = _usable_on_rows(frame)
    if work.empty:
        return []
    session_ids = work["_time"].diff().gt(timedelta(minutes=SESSION_GAP_MINUTES)).cumsum()
    work = work.loc[session_ids.eq(session_ids.iloc[-1])]
    if len(work) < 30:
        return []
    ranges = _approved_ranges(profile)
    warnings = []
    reference = work.iloc[:-SESSION_MAX_ROWS] if len(work) > SESSION_MAX_ROWS + 20 else work
    for field, (approved_low, approved_high) in ranges.items():
        values = pd.to_numeric(reference.get(field), errors="coerce").dropna()
        actual = pd.to_numeric(reading.get(field), errors="coerce")
        if len(values) < 20 or pd.isna(actual):
            continue
        actual = float(actual)
        if not float(approved_low) <= actual <= float(approved_high):
            continue
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = max(1.4826 * mad, ROBUST_SCALE_FLOORS[field])
        z_score = (actual - median) / scale
        if abs(z_score) < minimum_z:
            continue
        direction = "HIGH" if z_score > 0 else "LOW"
        if field.startswith("pf_") and direction == "HIGH":
            # A rising power factor is an improvement, not a degradation signal.
            continue
        warnings.append({
            "code": f"UNUSUAL_{field.upper()}_{direction}",
            "label": f"Sudden change in {field_name(field)[0].lower() + field_name(field)[1:]}",
            "field": field,
            "direction": direction,
            "current_value": round(actual, 3),
            "recent_median": round(median, 3),
            "robust_scale": round(scale, 3),
            "robust_z_score": round(float(z_score), 2),
            "detection_source": "RECENT_ROBUST_CONTEXT",
            "evidence": (
                f"{field_name(field)} {'rose' if direction == 'HIGH' else 'dropped'} to {with_unit(actual, field)} "
                f"during this run; earlier in the run it was steady around {with_unit(median, field)}. "
                "Still within the safe range."
            ),
        })
    return warnings


LONG_TERM_FIELDS = (
    "current_r", "current_y", "current_b", "voltage_r", "voltage_y", "voltage_b",
    "pf_r", "pf_y", "pf_b", "total_kw",
)


def long_term_normal(hourly: pd.DataFrame, *, days: int = 7, min_hours: int = 24) -> dict | None:
    """Median of each field over the pump's ON hours in the last ``days`` days.

    The median ignores short faults, spikes and start-up transients that would
    pull an average towards abnormal values.
    """
    if hourly is None or hourly.empty or "_time" not in hourly:
        return None
    work = hourly.copy()
    work["_time"] = pd.to_datetime(work["_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["_time"])
    if "usable_on_samples" in work:
        work = work.loc[pd.to_numeric(work["usable_on_samples"], errors="coerce").fillna(0) > 0]
    if work.empty:
        return None
    work = work.loc[work["_time"] >= work["_time"].max() - timedelta(days=int(days))]
    if len(work) < min_hours:
        return None
    medians = {}
    for field in LONG_TERM_FIELDS:
        if field not in work:
            continue
        values = pd.to_numeric(work[field], errors="coerce").dropna()
        if len(values) >= min_hours:
            medians[field] = float(values.median())
    if not medians:
        return None
    return {"medians": medians, "hours": int(len(work)), "days": int(days),
            "period_start": work["_time"].min().isoformat(), "period_end": work["_time"].max().isoformat()}


def long_term_shift_warnings(
    reading: pd.Series | None,
    normal: dict | None,
    profile: dict,
    *,
    percent: float = 15.0,
) -> list[dict]:
    """Flag a run whose level is far from the pump's usual level, while still inside approved limits.

    This catches a pump that starts a run already shifted, which the in-session
    check cannot see because that run has no earlier readings to compare with.
    """
    if reading is None or not normal:
        return []
    medians = normal.get("medians") or {}
    days = normal.get("days", 7)
    warnings = []
    for field, (approved_low, approved_high) in _approved_ranges(profile).items():
        usual = medians.get(field)
        actual = pd.to_numeric(reading.get(field), errors="coerce")
        if usual is None or pd.isna(actual) or abs(float(usual)) < 1e-9:
            continue
        actual = float(actual)
        if not float(approved_low) <= actual <= float(approved_high):
            continue
        change = (actual - float(usual)) / abs(float(usual)) * 100.0
        if abs(change) < percent:
            continue
        direction = "HIGH" if change > 0 else "LOW"
        if field.startswith("pf_") and direction == "HIGH":
            continue  # a higher power factor is an improvement
        warnings.append({
            "code": f"UNUSUAL_{field.upper()}_{direction}",
            "label": f"Unusual {field_name(field)[0].lower() + field_name(field)[1:]}",
            "field": field,
            "direction": direction,
            "current_value": round(actual, 3),
            "long_term_median": round(float(usual), 3),
            "change_percent": round(change, 1),
            "detection_source": "LONG_TERM_NORMAL",
            "evidence": (
                f"{field_name(field)} is {with_unit(actual, field)}, {abs(change):.0f}% {'above' if change > 0 else 'below'} "
                f"this pump's usual {with_unit(usual, field)} (typical over the last {count(days, 'day')}). "
                "Still within the safe range."
            ),
        })
    return warnings


def immediate_faults(latest: pd.Series, profile: dict) -> list[dict]:
    """Keep phase-loss evidence immediate; median-confirm the remaining rules."""
    from backend.engine.electrical_profile import current_faults

    immediate_codes = {"PHASE_LOSS", "VOLTAGE_PHASE_LOSS"}
    return [fault for fault in current_faults(latest, profile) if fault["code"] in immediate_codes]

