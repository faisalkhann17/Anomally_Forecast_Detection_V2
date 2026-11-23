"""Corroborated meter-only slow-degradation patterns and maintenance priority."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.engine.pump_performance_features import build_performance_features
from backend.engine.readable_text import count, field_name, with_unit


PRIORITY_WEIGHTS = {
    "POSSIBLE_HYDRAULIC_RESTRICTION": 18,
    "PUMP_EFFICIENCY_DEGRADATION": 20,
    "SOURCE_STARVATION_RISK": 20,
    "MOTOR_ELECTRICAL_DETERIORATION": 20,
    "POSSIBLE_SUPPLY_SIDE_DEGRADATION": 15,
    "POSSIBLE_HYDRAULIC_WEAR_OR_FOULING": 18,
    "ABNORMAL_OPERATING_POINT_DRIFT": 12,
}


# Patterns that describe the same physical problem share a group, so one root
# cause (for example a flow drop) cannot count as several independent issues.
PATTERN_GROUPS = {
    "PUMP_EFFICIENCY_DEGRADATION": "HYDRAULIC",
    "POSSIBLE_HYDRAULIC_RESTRICTION": "HYDRAULIC",
    "POSSIBLE_HYDRAULIC_WEAR_OR_FOULING": "HYDRAULIC",
    "ABNORMAL_OPERATING_POINT_DRIFT": "HYDRAULIC",
    "MOTOR_ELECTRICAL_DETERIORATION": "MOTOR",
    "POSSIBLE_SUPPLY_SIDE_DEGRADATION": "SUPPLY",
    "SOURCE_STARVATION_RISK": "WATER_SOURCE",
}
# Single measured trends that are not yet part of a pattern, by the group they belong to.
TREND_GROUPS = {
    "LOAD_CREEP": "MOTOR",
    "PF_DEGRADATION": "MOTOR",
    "PHASE_ASYMMETRY_TREND": "MOTOR",
    "SUPPLY_QUALITY_RISK": "SUPPLY",
    "ENERGY_INTENSITY_DETERIORATION": "HYDRAULIC",
    "FLOW_PER_POWER_DETERIORATION": "HYDRAULIC",
    "DELIVERY_RUNTIME_DETERIORATION": "HYDRAULIC",
}
GROUP_LABELS = {
    "HYDRAULIC": "Hydraulic / pump efficiency",
    "MOTOR": "Motor / electrical",
    "SUPPLY": "Power supply",
    "WATER_SOURCE": "Water source",
}
# Maintenance is recommended only when this many independent issue groups develop together.
MAINTENANCE_MIN_GROUPS = 2
URGENT_DAYS = 7
CONFIDENCE_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
LEVEL_BY_GUIDANCE = {
    "NORMAL": "NORMAL",
    "DEVELOPING_ISSUES": "DEVELOPING_ISSUES",
    "PLAN_INSPECTION": "MAINTENANCE_ATTENTION",
    "URGENT_INSPECTION": "HIGH_PRIORITY",
}


def priority_level(score: int) -> str:
    """Legacy score band; the operator level now comes from issue groups (see _maintenance_guidance)."""
    if score <= 30:
        return "NORMAL"
    if score <= 50:
        return "OBSERVE"
    if score <= 75:
        return "MAINTENANCE_ATTENTION"
    return "HIGH_PRIORITY"


def _readable(code: str) -> str:
    """Turn a signal code into a short sentence-case label."""
    return code.replace("_", " ").capitalize()


def _developing_issues(signals: list[dict], electrical: dict, performance: dict) -> list[dict]:
    """List every slowly developing issue: combined patterns, then trends not yet part of a pattern."""
    used = {factor for signal in signals for factor in signal.get("supporting_signals", [])}
    issues = []
    for signal in signals:
        group = PATTERN_GROUPS.get(signal["code"], "OTHER")
        issues.append({
            "kind": "PATTERN", "code": signal["code"], "label": signal.get("label") or _readable(signal["code"]),
            "group": group, "group_label": GROUP_LABELS.get(group, "Other"),
            "evidence": signal.get("evidence"), "supporting_signals": signal.get("supporting_signals", []),
            "confidence": signal.get("confidence"), "estimated_days_to_threshold": None,
        })
    for trend in electrical.get("signals", []):
        code = str(trend.get("code") or "")
        if not code or code in used:
            continue
        group = TREND_GROUPS.get(code, "OTHER")
        current, baseline = trend.get("current_value"), trend.get("baseline_median")
        evidence = None
        if current is not None and baseline is not None:
            metric = str(trend.get("metric") or code)
            evidence = f"{field_name(metric)}: was {with_unit(baseline, metric)}, now {with_unit(current, metric)}."
            if trend.get("threshold") is not None:
                evidence += f" Safe limit is {with_unit(trend['threshold'], metric)}."
        issues.append({
            "kind": "TREND", "code": code, "label": trend.get("label") or _readable(code),
            "group": group, "group_label": GROUP_LABELS.get(group, "Other"), "evidence": evidence,
            "supporting_signals": [], "confidence": electrical.get("confidence"),
            "estimated_days_to_threshold": trend.get("estimated_days_to_threshold"),
        })
    for trend in performance.get("signals", []):
        code = str(trend.get("code") or "")
        if not code or code in used:
            continue
        group = TREND_GROUPS.get(code, "OTHER")
        change = trend.get("change_percent")
        evidence = (trend.get("evidence") or "") + ("" if change is None else f" ({float(change):+.1f}% compared with earlier running)")
        issues.append({
            "kind": "TREND", "code": code, "label": _readable(code), "group": group,
            "group_label": GROUP_LABELS.get(group, "Other"), "evidence": evidence.strip(),
            "supporting_signals": [], "confidence": performance.get("confidence"), "estimated_days_to_threshold": None,
        })
    return issues


def _codes(payload: dict) -> set[str]:
    """Extract signal codes from an analysis payload for corroboration."""
    return {str(signal.get("code")) for signal in payload.get("signals", []) if signal.get("code")}


def _change(changes: dict, field: str) -> float | None:
    """Read a finite percentage-change value from a feature dictionary."""
    value = changes.get(field)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _source_changes(frame: pd.DataFrame, comparison_hours: int = 168) -> dict[str, float | None]:
    """Compare recent source level, flow, power and current against history."""
    if frame.empty or "_time" not in frame:
        return {"borewell_water_level": None}
    work = frame.copy()
    work["_time"] = pd.to_datetime(work["_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["_time"]).set_index("_time").sort_index()
    status_on = work.get("pump_status", pd.Series("UNKNOWN", index=work.index)).astype(str).str.upper().isin(["ON", "RUNNING"])
    currents = work.reindex(columns=["current_r", "current_y", "current_b"]).apply(pd.to_numeric, errors="coerce")
    power = pd.to_numeric(work.get("total_kw", pd.Series(np.nan, index=work.index)), errors="coerce")
    physical_on = currents.max(axis=1).gt(0.5) | power.gt(0.1)
    usable_on = status_on & physical_on
    hourly = pd.DataFrame(index=work.resample("1h").size().index)
    for field in ("borewell_water_level", "flowrate", "total_kw"):
        values = pd.to_numeric(work.get(field, pd.Series(np.nan, index=work.index)), errors="coerce")
        hourly[field] = values.where(usable_on).resample("1h").mean()
    hourly["current_avg"] = currents.mean(axis=1).where(usable_on).resample("1h").mean()
    hourly = hourly.dropna(how="all")
    effective = min(comparison_hours, len(hourly) // 2)
    if effective < 24:
        return {field: None for field in hourly.columns}
    reference = hourly.iloc[-effective * 2:-effective]
    recent = hourly.tail(effective)
    changes = {}
    for field in hourly.columns:
        old = reference[field].median(skipna=True)
        new = recent[field].median(skipna=True)
        changes[field] = None if not np.isfinite(old) or not np.isfinite(new) or abs(old) < 1e-9 else float((new - old) / abs(old) * 100)
    return changes


def _confidence(electrical: dict, performance: dict, *, fleet: bool = False) -> str:
    """Combine evidence confidence conservatively across analysis layers."""
    if fleet:
        return "HIGH"
    rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    evidence_confidences = []
    if electrical.get("signals"):
        evidence_confidences.append(rank.get(str(electrical.get("confidence", "NONE")), 0))
    if performance.get("signals"):
        evidence_confidences.append(rank.get(str(performance.get("confidence", "NONE")), 0))
    if not evidence_confidences:
        return "LOW"
    lowest = min(evidence_confidences)
    return {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}[lowest]


def _threshold_window(raw_history: pd.DataFrame, profile: dict | None) -> dict | None:
    """Estimate a robust advisory window to the engineer's minimum delivery flow.

    Energy-per-m³, flow-per-kW and runtime-per-m³ limits were removed: engineers
    cannot supply them, so those metrics are judged only by their ±10% trend.
    """
    limits = (profile or {}).get("operating_limits", {})
    specifications = (
        ("flowrate", "flowrate_min", "decreasing"),
    )
    hourly = build_performance_features(raw_history)
    if hourly.empty:
        return None
    daily = hourly.resample("1D").median(numeric_only=True)
    candidates = []
    for metric, limit_key, direction in specifications:
        if limits.get(limit_key) is None or metric not in daily:
            continue
        values = daily[metric].dropna().tail(45)
        if len(values) < 7:
            continue
        day_numbers = (values.index - values.index[0]).total_seconds().to_numpy() / 86400
        pairwise_slopes = []
        for left in range(len(values) - 1):
            elapsed = day_numbers[left + 1:] - day_numbers[left]
            pairwise_slopes.extend(((values.iloc[left + 1:].to_numpy() - values.iloc[left]) / elapsed).tolist())
        slopes = np.asarray([slope for slope in pairwise_slopes if np.isfinite(slope)])
        expected = slopes[slopes > 0] if direction == "increasing" else -slopes[slopes < 0]
        if len(expected) < max(6, int(len(slopes) * 0.7)):
            continue
        magnitude_low, magnitude_mid, magnitude_high = np.quantile(expected, [0.25, 0.5, 0.75])
        current = float(values.tail(3).median())
        threshold = float(limits[limit_key])
        distance = threshold - current if direction == "increasing" else current - threshold
        if distance <= 0:
            days_low = days_mid = days_high = 0.0
        elif magnitude_low <= 1e-9:
            continue
        else:
            days_low = distance / magnitude_high
            days_mid = distance / magnitude_mid
            days_high = distance / magnitude_low
        if days_mid > 90:
            continue
        candidates.append({
            "metric": metric,
            "threshold_source": "ENGINEER_APPROVED",
            "current_value": round(current, 4),
            "threshold": round(threshold, 4),
            "slope_per_day": round(magnitude_mid if direction == "increasing" else -magnitude_mid, 6),
            "estimated_days_to_threshold": round(max(0, days_mid), 1),
            "estimated_window_days": [round(max(0, days_low), 1), round(max(0, days_high), 1)],
            "daily_points": len(values),
        })
    return min(candidates, key=lambda item: item["estimated_days_to_threshold"]) if candidates else None


def _maintenance_guidance(signals: list[dict], confidence: str, threshold_window: dict | None,
                          developing_issues: list[dict] | None = None, electrical_days: float | None = None) -> dict:
    """Recommend maintenance only when independent issue groups develop together.

    * no issue                          -> NORMAL
    * issues in fewer than 2 groups     -> DEVELOPING_ISSUES (shown as an alert, no maintenance call)
    * 2+ groups with MEDIUM/HIGH trust  -> PLAN_INSPECTION, or URGENT_INSPECTION when a limit is
                                           expected within URGENT_DAYS
    """
    developing_issues = developing_issues or []
    groups = sorted({PATTERN_GROUPS.get(signal["code"], "OTHER") for signal in signals})
    days_candidates = [value for value in (
        None if threshold_window is None else threshold_window.get("estimated_days_to_threshold"), electrical_days,
    ) if value is not None]
    earliest = min(days_candidates) if days_candidates else None
    base = {
        "issue_groups": groups,
        "issue_group_labels": [GROUP_LABELS.get(group, "Other") for group in groups],
        "supporting_factor_count": len(groups),
        "supporting_factors": groups,
        "minimum_groups_for_maintenance": MAINTENANCE_MIN_GROUPS,
        "inspection_window_days": None,
        "estimated_days_to_threshold": earliest,
    }
    if not developing_issues:
        return {**base, "state": "NORMAL", "action": "No maintenance action required",
                "reason": "No slowly developing issue was found."}
    if len(groups) < MAINTENANCE_MIN_GROUPS or CONFIDENCE_RANK.get(confidence, 0) < 2:
        reason = (f"Tracking {count(len(developing_issues), 'slowly developing issue')}. A service visit is suggested "
                  f"when problems show up in at least {MAINTENANCE_MIN_GROUPS} separate areas.")
        if len(groups) >= MAINTENANCE_MIN_GROUPS:
            reason = "Issues are developing in several groups, but more reliable history is needed before recommending maintenance."
        return {**base, "state": "DEVELOPING_ISSUES", "action": "Track these developing issues", "reason": reason}
    urgent = earliest is not None and earliest <= URGENT_DAYS
    return {
        **base,
        "state": "URGENT_INSPECTION" if urgent else "PLAN_INSPECTION",
        "action": ("Arrange an engineering inspection now" if earliest is not None and earliest <= 0 else
                   "Arrange an engineering inspection soon" if urgent else "Plan an engineering inspection"),
        "reason": f"Problems are showing up in {len(groups)} separate areas: {', '.join(base['issue_group_labels'])}.",
        "inspection_window_days": None if threshold_window is None else threshold_window.get("estimated_window_days"),
    }


def analyze_combined_slow_degradation(electrical: dict, performance: dict, raw_history: pd.DataFrame, profile: dict | None = None) -> dict:
    """Corroborate meter trends into possible causes and inspection priority.

    Possible wear, restriction or source issues are hypotheses, not confirmed
    mechanical diagnoses or failure probabilities.
    """
    electrical_codes = _codes(electrical)
    performance_codes = _codes(performance)
    changes = performance.get("changes_percent", {})
    source_changes = _source_changes(raw_history)
    confidence = _confidence(electrical, performance)
    signals: list[dict] = []

    energy_change = _change(changes, "energy_per_m3")
    flow_efficiency_change = _change(changes, "flow_per_kw")
    runtime_change = _change(changes, "runtime_per_m3")
    power_change = _change(changes, "total_kw")
    flow_change = _change(changes, "flowrate")
    current_change = _change(changes, "current_avg")
    frequency_change = _change(changes, "frequency")

    efficiency_evidence = {
        "ENERGY_INTENSITY_DETERIORATION",
        "FLOW_PER_POWER_DETERIORATION",
        "DELIVERY_RUNTIME_DETERIORATION",
    } & performance_codes
    if len(efficiency_evidence) >= 2:
        signals.append({
            "code": "PUMP_EFFICIENCY_DEGRADATION",
            "label": "Pump efficiency degradation",
            "confidence": confidence,
            "severity": 0.75 if len(efficiency_evidence) == 3 else 0.6,
            "evidence": "The pump is using more energy and running longer for the same amount of water.",
            "supporting_signals": sorted(efficiency_evidence),
        })

    deteriorating_delivery = (flow_change is not None and flow_change <= -10) or (flow_efficiency_change is not None and flow_efficiency_change <= -10)
    energy_worse = energy_change is not None and energy_change >= 10
    loading_change = max(value for value in (power_change, current_change) if value is not None) if any(value is not None for value in (power_change, current_change)) else None
    if energy_worse and deteriorating_delivery and loading_change is not None:
        if loading_change > 5:
            signals.append({
                "code": "POSSIBLE_HYDRAULIC_RESTRICTION",
                "label": "Possible hydraulic restriction",
                "confidence": confidence,
                "severity": 0.7,
                "evidence": "Water flow fell while the motor worked harder and used more energy per m³. Something may be blocking the flow.",
                "supporting_signals": sorted(efficiency_evidence),
            })
        elif abs(loading_change) <= 8:
            signals.append({
                "code": "POSSIBLE_HYDRAULIC_WEAR_OR_FOULING",
                "label": "Possible hydraulic wear or fouling",
                "confidence": confidence,
                "severity": 0.65,
                "evidence": "The pump delivers less water for the same power. The impeller may be worn or dirty.",
                "supporting_signals": sorted(efficiency_evidence),
            })

    source_level_change = source_changes.get("borewell_water_level")
    source_flow_change = source_changes.get("flowrate")
    source_load_change = max(value for value in (source_changes.get("total_kw"), source_changes.get("current_avg")) if value is not None) if any(value is not None for value in (source_changes.get("total_kw"), source_changes.get("current_avg"))) else None
    if source_level_change is not None and source_level_change <= -5 and source_flow_change is not None and source_flow_change <= -10 and source_load_change is not None and source_load_change <= -5:
        signals.append({
            "code": "SOURCE_STARVATION_RISK",
            "label": "Source starvation risk",
            "confidence": confidence,
            "severity": 0.75,
            "evidence": "Water level, flow and motor load all fell together. The water source may be running low.",
            "supporting_signals": ["BOREWELL_LEVEL_DECLINE", "FLOW_DECLINE", "LOAD_DECLINE"],
        })

    motor_evidence = {"LOAD_CREEP", "PF_DEGRADATION", "PHASE_ASYMMETRY_TREND"} & electrical_codes
    if len(motor_evidence) >= 2:
        signals.append({
            "code": "MOTOR_ELECTRICAL_DETERIORATION",
            "label": "Motor or electrical deterioration",
            "confidence": str(electrical.get("confidence", "LOW")),
            "severity": 0.8,
            "evidence": "Two or more motor readings (load, power factor, phase balance) are getting worse together.",
            "supporting_signals": sorted(motor_evidence),
        })

    supply_evidence = {"SUPPLY_QUALITY_RISK", "PHASE_ASYMMETRY_TREND"} & electrical_codes
    if len(supply_evidence) == 2:
        signals.append({
            "code": "POSSIBLE_SUPPLY_SIDE_DEGRADATION",
            "label": "Possible supply-side degradation",
            "confidence": str(electrical.get("confidence", "LOW")),
            "severity": 0.65,
            "evidence": "Supply voltage is becoming uneven and phase currents are becoming unbalanced at the same time.",
            "supporting_signals": sorted(supply_evidence),
            "fleet_corroborated": False,
        })

    stable_frequency = frequency_change is not None and abs(frequency_change) <= 1
    if stable_frequency and flow_efficiency_change is not None and flow_efficiency_change <= -10 and deteriorating_delivery:
        signals.append({
            "code": "ABNORMAL_OPERATING_POINT_DRIFT",
            "label": "Abnormal operating-point drift",
            "confidence": confidence,
            "severity": 0.55,
            "evidence": "The pump now gives a different amount of water for the same power, at the same speed.",
            "supporting_signals": ["FLOW_PER_POWER_DETERIORATION", "STABLE_FREQUENCY"],
        })

    base_score = min(100, round(sum(PRIORITY_WEIGHTS[signal["code"]] * float(signal["severity"]) for signal in signals)))
    threshold_window = _threshold_window(raw_history, profile) if signals else None
    developing = _developing_issues(signals, electrical, performance)
    if threshold_window:
        for issue in developing:
            if issue["group"] == "HYDRAULIC":
                issue["estimated_days_to_threshold"] = threshold_window["estimated_days_to_threshold"]
    electrical_days = electrical.get("estimated_days_to_threshold")
    guidance = _maintenance_guidance(signals, confidence, threshold_window, developing, electrical_days)
    level = LEVEL_BY_GUIDANCE[guidance["state"]]
    actionable_window = guidance["state"] in {"PLAN_INSPECTION", "URGENT_INSPECTION"}
    if threshold_window and actionable_window:
        for signal in signals:
            if signal["code"] in {
                "POSSIBLE_HYDRAULIC_RESTRICTION", "PUMP_EFFICIENCY_DEGRADATION",
                "POSSIBLE_HYDRAULIC_WEAR_OR_FOULING", "ABNORMAL_OPERATING_POINT_DRIFT",
            }:
                signal["threshold_forecast"] = threshold_window
    return {
        "state": level,
        "priority_score": base_score,
        "base_priority_score": base_score,
        "priority_level": level,
        "maintenance_required": actionable_window,
        "developing_issues": developing,
        "decision_inputs": {"confidence": confidence, "threshold_window": threshold_window, "electrical_days": electrical_days},
        "confidence": confidence if signals else "NONE",
        "estimated_days_to_threshold": guidance.get("estimated_days_to_threshold") if actionable_window else None,
        "estimated_maintenance_window_days": guidance.get("inspection_window_days") if actionable_window else None,
        "threshold_forecast": threshold_window,
        "maintenance_guidance": guidance,
        "signals": signals,
        "changes_percent": {
            **changes,
            "borewell_water_level": None if source_level_change is None else round(source_level_change, 2),
        },
        "limitations": "Pattern-based meter evidence only. Time windows require an engineer-approved minimum delivery flow and at least 7 daily trend points; they are threshold estimates, not failure dates.",
    }


def corroborate_common_supply(payloads: dict[str, dict]) -> None:
    """Raise supply confidence only when two or more pumps show the same pattern."""
    candidates = []
    for pump_id, payload in payloads.items():
        combined = payload.get("combined_slow_degradation", {})
        for signal in combined.get("signals", []):
            if signal.get("code") == "POSSIBLE_SUPPLY_SIDE_DEGRADATION":
                candidates.append((pump_id, combined, signal))
    corroborated = len(candidates) >= 2
    for pump_id, combined, signal in candidates:
        signal["fleet_corroborated"] = corroborated
        # Payloads persist between runs, so keep the pump's own wording and restore it
        # once the peers recover instead of leaving a stale fleet-wide label behind.
        original = signal.setdefault("uncorroborated", {
            key: signal.get(key) for key in ("label", "confidence", "evidence")
        })
        if corroborated:
            peers = sorted(candidate_id for candidate_id, _, _ in candidates if candidate_id != pump_id)
            signal["label"] = "Supply-side common problem"
            signal["confidence"] = "HIGH"
            signal["evidence"] = f"The same voltage and phase trend is seen on {count(len(peers), 'other pump')}, so the cause is likely the power supply."
            signal["corroborating_pumps"] = peers
        else:
            signal.update(original)
            signal.pop("corroborating_pumps", None)
        score = min(100, int(combined.get("base_priority_score", 0)) + (10 if corroborated else 0))
        combined["priority_score"] = score
        inputs = combined.get("decision_inputs") or {}
        guidance = _maintenance_guidance(combined.get("signals", []), inputs.get("confidence", combined.get("confidence", "NONE")),
                                         inputs.get("threshold_window"), combined.get("developing_issues", []), inputs.get("electrical_days"))
        combined["maintenance_guidance"] = guidance
        combined["priority_level"] = combined["state"] = LEVEL_BY_GUIDANCE[guidance["state"]]
        combined["maintenance_required"] = guidance["state"] in {"PLAN_INSPECTION", "URGENT_INSPECTION"}
        for issue in combined.get("developing_issues", []):
            if issue.get("code") == "POSSIBLE_SUPPLY_SIDE_DEGRADATION":
                issue.update({key: signal.get(key) for key in ("label", "evidence", "confidence")})
