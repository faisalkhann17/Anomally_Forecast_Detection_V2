"""Slow electrical-degradation analysis using hourly energy-meter telemetry only."""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from backend.engine.electrical_profile import PHASES, current_faults, enriched_frame


MAX_THRESHOLD_ESTIMATE_DAYS = 90
MINIMUM_ADAPTIVE_HOURLY_POINTS = 48
MEDIUM_CONFIDENCE_HOURLY_POINTS = 120
TARGET_HOURLY_POINTS = 240


def _imbalance(values: np.ndarray) -> np.ndarray:
    """Calculate row-wise phase imbalance for maintenance trending."""
    average = np.nanmean(values, axis=1)
    return np.where(average > 0, np.nanmax(np.abs(values - average[:, None]), axis=1) / average * 100, np.nan)


def _mad(values: np.ndarray) -> float:
    """Calculate a robust median absolute deviation scale."""
    median = float(np.nanmedian(values))
    return float(max(np.nanmedian(np.abs(values - median)) * 1.4826, 1e-6))


def _theil_sen(values: np.ndarray) -> float:
    """Estimate a robust trend slope from pairwise median slopes."""
    pairs = [(values[right] - values[left]) / (right - left) for left, right in combinations(range(len(values)), 2) if np.isfinite(values[left]) and np.isfinite(values[right])]
    return float(np.median(pairs)) if pairs else 0.0


def _metric_payload(values: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarize health features with robust level and spread values."""
    result = {}
    for name in values.columns:
        data = values[name].to_numpy(dtype=float)
        result[name] = {"median": round(float(np.nanmedian(data)), 6), "spread": round(_mad(data), 6)}
    return result


def _feature_frame(frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """Create clean hourly electrical features for degradation rules."""
    if frame is None or frame.empty or "_time" not in frame.columns:
        # No data in the history window (e.g. a pump that stopped reporting): nothing to learn yet.
        return pd.DataFrame()
    work = enriched_frame(frame).copy()
    work["_time"] = pd.to_datetime(work["_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["_time"]).sort_values("_time")
    current_columns = [f"current_{phase}" for phase in PHASES]
    voltage_columns = [f"voltage_{phase}" for phase in PHASES]
    pf_columns = [f"pf_{phase}" for phase in PHASES]
    numeric = current_columns + voltage_columns + pf_columns + ["total_kw"]
    for name in numeric:
        work[name] = pd.to_numeric(work[name], errors="coerce")
    work["physical_on"] = (work[current_columns].mean(axis=1) > .5) | (work["total_kw"] > .1)
    work = work[work["physical_on"] & work[numeric].notna().all(axis=1)].copy()
    if work.empty:
        return work
    work["current_avg"] = work[current_columns].mean(axis=1)
    work["pf_avg"] = work[pf_columns].mean(axis=1)
    work["current_imbalance_pct"] = _imbalance(work[current_columns].to_numpy(dtype=float))
    work["voltage_imbalance_pct"] = _imbalance(work[voltage_columns].to_numpy(dtype=float))
    work["has_current_fault"] = work.apply(lambda row: bool(current_faults(row, profile)), axis=1)
    work = work[~work["has_current_fault"]]
    if work.empty:
        return work
    return work.set_index("_time")[["current_avg", "total_kw", "pf_avg", "current_imbalance_pct", "voltage_imbalance_pct"]].resample("1h").mean().dropna()


def build_baseline(hourly: pd.DataFrame, points: int) -> dict | None:
    """Build the pump's robust comparison baseline from early ON hours."""
    sample = hourly.iloc[:points]
    if len(sample) < points:
        return None
    return {"period_start": sample.index.min().isoformat(), "period_end": sample.index.max().isoformat(), "row_count": len(sample), "metrics": _metric_payload(sample)}


def _adaptive_windows(total_points: int, baseline_points: int, recent_points: int, persistence_points: int) -> tuple[int, int, int]:
    """Scale analysis windows to the amount of usable history."""
    if total_points >= baseline_points + recent_points:
        return baseline_points, recent_points, persistence_points
    recent = max(18, min(recent_points, total_points // 3))
    baseline = total_points - recent
    persistence = max(4, min(persistence_points, recent // 3))
    return baseline, recent, persistence


def _history_confidence(points: int) -> tuple[str, str]:
    """Map available hourly history to confidence and learning stage."""
    if points >= TARGET_HOURLY_POINTS:
        return "HIGH", "MATURE"
    if points >= MEDIUM_CONFIDENCE_HOURLY_POINTS:
        return "MEDIUM", "DEVELOPING"
    return "LOW", "EARLY_LEARNING"


def _signal(code: str, label: str, metric: str, direction: str, current: float, baseline: dict, threshold: float, slope: float, *, days: float | None, corroborated: bool) -> dict:
    """Build one explainable slow-degradation advisory signal."""
    severity = abs(current - baseline["median"]) / max(baseline["spread"] * 3, 1e-6)
    return {
        "code": code, "label": label, "metric": metric, "direction": direction,
        "current_value": round(current, 4), "baseline_median": baseline["median"], "baseline_spread": baseline["spread"],
        "threshold": round(threshold, 4), "slope_per_day": round(slope, 5),
        "estimated_days_to_threshold": None if days is None else round(max(days, 0), 1),
        "severity": round(float(min(1, severity / 2)), 3), "timestral_corroborated": corroborated,
    }


def analyze_electrical_maintenance(hourly_history: pd.DataFrame, profile: dict, baseline: dict | None, *, baseline_points: int = 168, recent_points: int = 72, persistence_points: int = 12, forecast_risk: dict | None = None) -> tuple[dict, dict | None]:
    """Compare sustained hourly electrical trends with this pump's earlier operation.

    Returns advisory degradation signals and the stored comparison baseline;
    it does not forecast a confirmed failure date.
    """
    hourly = _feature_frame(hourly_history, profile)
    if len(hourly) < MINIMUM_ADAPTIVE_HOURLY_POINTS:
        return {"state": "INSUFFICIENT_HISTORY", "required_hourly_points": MINIMUM_ADAPTIVE_HOURLY_POINTS, "target_hourly_points": TARGET_HOURLY_POINTS, "usable_hourly_points": len(hourly), "learning_stage": "LEARNING", "confidence": "NONE", "signals": []}, baseline
    effective_baseline, effective_recent, effective_persistence = _adaptive_windows(
        len(hourly), baseline_points, recent_points, persistence_points
    )
    created = False
    if baseline is None or int(baseline.get("row_count", 0)) < effective_baseline:
        baseline = build_baseline(hourly, effective_baseline)
        if baseline is None:
            return {"state": "INSUFFICIENT_HISTORY", "usable_hourly_points": len(hourly), "signals": []}, None
        created = True
    recent = hourly.tail(effective_recent)
    limits = profile["operating_limits"]
    definitions = (
        ("LOAD_CREEP", "Electrical load creep", "current_avg", "HIGH", float(limits["current_max_a"]), "RISING_LOAD_TREND"),
        ("PF_DEGRADATION", "Power factor degradation", "pf_avg", "LOW", float(limits["power_factor_min"]), "POWER_FACTOR_DEGRADING"),
        ("PHASE_ASYMMETRY_TREND", "Phase asymmetry trend", "current_imbalance_pct", "HIGH", float(limits["current_imbalance_max_percent"]), "PHASE_DIVERGENCE_TREND"),
        ("SUPPLY_QUALITY_RISK", "Supply quality trend", "voltage_imbalance_pct", "HIGH", float(limits["voltage_imbalance_max_percent"]), "VOLTAGE_STABILITY_DEGRADING"),
    )
    forecast_codes = {item.get("code") for item in (forecast_risk or {}).get("signals", [])}
    signals = []
    for code, label, metric, direction, threshold, forecast_code in definitions:
        baseline_metric = (baseline.get("metrics") or {}).get(metric)
        if not baseline_metric:
            continue
        values = recent[metric].to_numpy(dtype=float)
        current, slope = float(np.nanmedian(values[-effective_persistence:])), _theil_sen(values) * 24
        median, spread = float(baseline_metric["median"]), max(float(baseline_metric["spread"]), 1e-6)
        trigger = median + 3 * spread if direction == "HIGH" else median - 3 * spread
        shifted = current > trigger if direction == "HIGH" else current < trigger
        rising = slope > spread / 30 if direction == "HIGH" else slope < -spread / 30
        persisted = np.all(values[-effective_persistence:] > trigger) if direction == "HIGH" else np.all(values[-effective_persistence:] < trigger)
        if not (shifted and persisted):
            continue
        crosses = (threshold - current) / slope if rising and ((direction == "HIGH" and slope > 0) or (direction == "LOW" and slope < 0)) else None
        if crosses is not None and (crosses < 0 or crosses > MAX_THRESHOLD_ESTIMATE_DAYS):
            crosses = None
        signals.append(_signal(code, label, metric, direction, current, baseline_metric, threshold, slope, days=crosses, corroborated=forecast_code in forecast_codes))
    estimated = [item["estimated_days_to_threshold"] for item in signals if item["estimated_days_to_threshold"] is not None]
    earliest = min(estimated) if estimated else None
    state = "NORMAL" if not signals else "ELECTRICAL_THRESHOLD_FORECAST" if earliest is not None else "OBSERVE"
    history_confidence, learning_stage = _history_confidence(len(hourly))
    corroborated = any(item["timestral_corroborated"] for item in signals)
    confidence = history_confidence if signals else "NONE"
    if corroborated and confidence == "LOW":
        confidence = "MEDIUM"
    if not signals:
        recommendation = "Continue monitoring"
    elif confidence == "LOW":
        recommendation = "Continue monitoring; more operating history is needed before scheduling maintenance"
    # Maintenance itself is decided in slow_degradation_patterns from independent issue
    # groups; this layer only describes how close its own trends are to a limit.
    elif earliest is None or earliest > 30:
        recommendation = "Developing trend; keep tracking it"
    elif earliest > 7:
        recommendation = "Trend may reach the approved limit within 30 days"
    else:
        recommendation = "Trend may reach the approved limit within 7 days"
    return {"state": state, "confidence": confidence, "learning_stage": learning_stage, "recommendation": recommendation, "estimated_days_to_threshold": earliest, "assessment_horizon_days": MAX_THRESHOLD_ESTIMATE_DAYS, "signals": signals, "baseline_created": created, "usable_hourly_points": len(hourly), "baseline_hourly_points": effective_baseline, "recent_hourly_points": len(recent), "persistence_hourly_points": effective_persistence, "minimum_hourly_points": MINIMUM_ADAPTIVE_HOURLY_POINTS, "target_hourly_points": TARGET_HOURLY_POINTS, "limitations": "Electrical threshold estimate only; it is not a pump failure date. Low-history assessments remain advisory."}, baseline
