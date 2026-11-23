"""Slow pump-performance features derived from status, energy, flow, and levels."""
from __future__ import annotations

import numpy as np
import pandas as pd


MINIMUM_PERFORMANCE_HOURLY_POINTS = 48
TARGET_PERFORMANCE_HOURLY_POINTS = 336


def _numeric(frame: pd.DataFrame, field: str) -> pd.Series:
    """Read one telemetry column as numeric while preserving missing values."""
    return pd.to_numeric(frame.get(field, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _on_state(frame: pd.DataFrame) -> pd.Series:
    """Convert authoritative pump status tags into a boolean ON series."""
    return frame.get("pump_status", pd.Series("unknown", index=frame.index)).astype(str).str.upper().isin(["ON", "RUNNING"])


def build_performance_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build hourly ON-session efficiency and delivery features."""
    work = frame.copy()
    work["_time"] = pd.to_datetime(work.get("_time"), utc=True, errors="coerce")
    work = work.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last")
    if work.empty:
        return pd.DataFrame()
    work["is_on"] = _on_state(work)
    work["total_kw"] = _numeric(work, "total_kw")
    work["flowrate"] = _numeric(work, "flowrate")
    work["total_kwh"] = _numeric(work, "total_kwh")
    work["totalizer"] = _numeric(work, "totalizer")
    work["frequency"] = _numeric(work, "frequency")
    currents = work.reindex(columns=["current_r", "current_y", "current_b"]).apply(pd.to_numeric, errors="coerce")
    work["current_avg"] = currents.mean(axis=1)
    interval_hours = work["_time"].diff().dt.total_seconds().div(3600)
    continuous_on = (
        work["is_on"]
        & work["is_on"].shift(fill_value=False)
        & interval_hours.gt(0)
        & interval_hours.le(1)
    )
    work["on_runtime_hours"] = interval_hours.where(continuous_on)
    work["energy_delta_kwh"] = work["total_kwh"].diff().where(continuous_on & work["total_kwh"].diff().gt(0))
    work["volume_delta_m3"] = work["totalizer"].diff().where(continuous_on & work["totalizer"].diff().gt(0))

    indexed = work.set_index("_time")
    on = indexed["is_on"]
    hourly = pd.DataFrame(index=indexed.resample("1h").size().index)
    for field in ("total_kw", "flowrate", "frequency", "current_avg"):
        hourly[field] = indexed[field].where(on).resample("1h").mean()
    hourly["energy_delta_kwh"] = indexed["energy_delta_kwh"].resample("1h").sum(min_count=1)
    hourly["volume_delta_m3"] = indexed["volume_delta_m3"].resample("1h").sum(min_count=1)
    hourly["on_runtime_hours"] = indexed["on_runtime_hours"].resample("1h").sum(min_count=1)
    hourly["flow_per_kw"] = (hourly["flowrate"] / hourly["total_kw"]).where(hourly["total_kw"] > 0)
    hourly["energy_per_m3"] = (hourly["energy_delta_kwh"] / hourly["volume_delta_m3"]).where(hourly["volume_delta_m3"] > 0)
    hourly["runtime_per_m3"] = (hourly["on_runtime_hours"] / hourly["volume_delta_m3"]).where(hourly["volume_delta_m3"] > 0)
    return hourly[["total_kw", "flowrate", "frequency", "current_avg", "energy_per_m3", "flow_per_kw", "runtime_per_m3"]]


def analyze_performance(frame: pd.DataFrame, *, comparison_hours: int = 168) -> dict:
    """Compare recent versus earlier flow/kW, kWh/m³ and runtime/m³.

    These measured-history changes require valid flow and cumulative counters;
    they are not direct Timestral predictions.
    """
    hourly = build_performance_features(frame).dropna(how="all")
    if len(hourly) < MINIMUM_PERFORMANCE_HOURLY_POINTS:
        return {"state": "INSUFFICIENT_HISTORY", "required_hourly_points": MINIMUM_PERFORMANCE_HOURLY_POINTS, "target_hourly_points": TARGET_PERFORMANCE_HOURLY_POINTS, "usable_hourly_points": len(hourly), "learning_stage": "LEARNING", "confidence": "NONE", "signals": []}
    effective_comparison = min(comparison_hours, len(hourly) // 2)
    reference, recent = hourly.iloc[-effective_comparison * 2:-effective_comparison], hourly.tail(effective_comparison)

    def metrics(values: pd.DataFrame) -> dict[str, float | None]:
        """Return recent/reference medians for operator efficiency KPIs."""
        result = {}
        for field in ("flow_per_kw", "energy_per_m3"):
            median = values[field].median(skipna=True)
            result[field] = round(float(median), 4) if np.isfinite(median) else None
        return result

    def change(field: str) -> float | None:
        """Calculate percentage change from reference to recent operation."""
        old, new = reference[field].median(skipna=True), recent[field].median(skipna=True)
        if not np.isfinite(old) or not np.isfinite(new) or abs(old) < 1e-9:
            return None
        return float((new - old) / abs(old) * 100)

    changes = {field: change(field) for field in (
        "energy_per_m3", "flow_per_kw", "runtime_per_m3",
        "total_kw", "flowrate", "current_avg", "frequency",
    )}
    signals = []
    if changes["energy_per_m3"] is not None and changes["energy_per_m3"] >= 10:
        signals.append({"code": "ENERGY_INTENSITY_DETERIORATION", "change_percent": round(changes["energy_per_m3"], 1), "evidence": "Energy used per m³ of water has gone up."})
    if changes["flow_per_kw"] is not None and changes["flow_per_kw"] <= -10:
        signals.append({"code": "FLOW_PER_POWER_DETERIORATION", "change_percent": round(changes["flow_per_kw"], 1), "evidence": "Water flow for each kW of power has gone down."})
    if changes["runtime_per_m3"] is not None and changes["runtime_per_m3"] >= 10:
        signals.append({"code": "DELIVERY_RUNTIME_DETERIORATION", "change_percent": round(changes["runtime_per_m3"], 1), "evidence": "The pump has to run longer to deliver each m³ of water."})
    pump_hydraulic = changes["energy_per_m3"] is not None and changes["energy_per_m3"] >= 10 and changes["flow_per_kw"] is not None and changes["flow_per_kw"] <= -10
    direction = "PUMP_OR_HYDRAULIC_PERFORMANCE" if pump_hydraulic else "UNCLASSIFIED"
    confidence = "HIGH" if len(hourly) >= TARGET_PERFORMANCE_HOURLY_POINTS else "MEDIUM" if len(hourly) >= 168 else "LOW"
    learning_stage = "MATURE" if confidence == "HIGH" else "DEVELOPING" if confidence == "MEDIUM" else "EARLY_LEARNING"
    return {"state": "MAINTENANCE_ATTENTION" if signals else "NORMAL", "confidence": confidence if signals else "NONE", "learning_stage": learning_stage, "comparison_hours": effective_comparison, "usable_hourly_points": len(hourly), "minimum_hourly_points": MINIMUM_PERFORMANCE_HOURLY_POINTS, "target_hourly_points": TARGET_PERFORMANCE_HOURLY_POINTS, "current_metrics": metrics(recent), "reference_metrics": metrics(reference), "changes_percent": {key: None if value is None else round(value, 2) for key, value in changes.items()}, "root_cause_direction": direction, "signals": signals, "limitations": "Efficiency signals require reliable cumulative energy and water-totalizer readings. Low-history assessments remain advisory."}


def rank_energy_waste(per_pump_features: dict[str, pd.DataFrame]) -> list[dict]:
    """Rank pumps by latest energy-per-volume, ignoring pumps without valid volume data."""
    rows = []
    for pump_id, hourly in per_pump_features.items():
        if "energy_per_m3" not in hourly:
            continue
        value = hourly["energy_per_m3"].dropna()
        if not value.empty:
            rows.append({"pump_id": pump_id, "energy_per_m3": round(float(value.iloc[-1]), 6)})
    return sorted(rows, key=lambda row: row["energy_per_m3"], reverse=True)
