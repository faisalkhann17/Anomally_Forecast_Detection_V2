"""Explainable, baseline-relative pump health scoring.

The calculation has four stages:

1. Convert each available sensor into a 0-100 channel score by measuring how
   far it is outside that pump's own expected range.
2. Use the worst channel score in each health group and combine the available
   groups using normalized weights.
3. Apply cross-group penalties and evidence-based caps/floors for confirmed
   faults, early warnings, and model-only evidence.
4. Optionally smooth the raw score over distinct telemetry timestamps and use
   hysteresis when the displayed health band recovers.

This module does not diagnose new faults. It projects existing detector,
baseline, data-quality, and short-cycling evidence into an operator-friendly
score without changing the authoritative detection state.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import math
from statistics import median
from typing import Any

def available_stat(stats: dict | None, field: str, required: tuple[str, ...] = ()) -> dict | None:
    """Return one usable baseline statistic only when required keys exist."""
    stat = (stats or {}).get(field)
    if not isinstance(stat, dict) or not stat.get("available", True):
        return None
    if any(key not in stat for key in required):
        return None
    return stat


def expected_range(spec: dict) -> tuple[float, float]:
    """Resolve a baseline's explicit or robustly derived expected range."""
    low = spec.get("expected_low")
    high = spec.get("expected_high")
    if low is not None and high is not None:
        return float(low), float(high)
    return (
        min(float(spec["p05"]), float(spec["median"]) - 3 * float(spec["scale"])),
        max(float(spec["p95"]), float(spec["median"]) + 3 * float(spec["scale"])),
    )


# Relative contribution of each evidence group. When one or more groups have
# no usable input, the remaining weights are normalized to sum to 1.0:
# effective_weight(group) = configured_weight(group) / sum(available weights)
GROUP_WEIGHTS = {
    "ELECTRICAL": 0.30,
    "LOAD": 0.35,
    "POWER_QUALITY": 0.20,
    "OPERATING": 0.15,
}

# Piecewise-linear mapping from normalized excess to channel health.
# normalized_excess = distance beyond expected boundary / baseline scale
# A reading inside its expected range has excess=0 and receives 100. Between
# breakpoints the score is linearly interpolated; excess >= 6 receives 0.
CHANNEL_BREAKPOINTS = (
    (0.0, 100.0),
    (1.0, 70.0),
    (3.0, 30.0),
    (6.0, 0.0),
)

# Fault severity controls both the affected group's clamp and the final score
# cap. Unknown fault codes intentionally fall back to the MAJOR tier.
CRITICAL_FAULTS = {
    "PHASE_LOSS",
    "VOLTAGE_PHASE_LOSS",
    "DRY_RUN",
    "MOTOR_OVERLOAD",
}
MAJOR_FAULTS = {
    "FLOW_RESTRICTION",
    "PHASE_CURRENT_IMBALANCE",
    "VOLTAGE_IMBALANCE",
    "UNDER_VOLTAGE",
    "OVER_VOLTAGE",
    "LOW_PF",
    "SHORT_CYCLING",
}

# Evidence safety limits. A cap prevents a serious active issue from appearing
# healthy even when the sensor-weighted score is high. An early warning alone
# is constrained to WATCH (60.0-79.9); it cannot create WARNING/CRITICAL.
CRITICAL_FAULT_CAP = 35.0
MAJOR_FAULT_CAP = 55.0
EARLY_WARNING_CAP = 79.9
EARLY_WARNING_FLOOR = 60.0
MODEL_EVIDENCE_CAP = 75.0
CRITICAL_GROUP_CLAMP = 10.0
MAJOR_GROUP_CLAMP = 40.0
# If more than one independently weighted group scores below 80, subtract five
# points for every additional low group, up to 15 points:
# penalty = min(15, 5 * max(0, low_group_count - 1))
MULTI_ISSUE_PENALTY_PER_EXTRA_GROUP = 5.0
MULTI_ISSUE_PENALTY_MAX = 15.0
LOW_GROUP_THRESHOLD = 80.0
# Median smoothing uses the latest five distinct telemetry timestamps. Band
# recovery needs three points above the next band's normal entry threshold;
# deterioration remains immediate.
HEALTH_SMOOTHING_WINDOW = 5
BAND_RECOVERY_HYSTERESIS = 3.0

# Inclusive score bands: GOOD >=80, WATCH >=60, WARNING >=40, else CRITICAL.
BAND_BOUNDARIES = (
    (80.0, "GOOD"),
    (60.0, "WATCH"),
    (40.0, "WARNING"),
    (0.0, "CRITICAL"),
)
BAND_ORDER = ("CRITICAL", "WARNING", "WATCH", "GOOD")
BAND_MINIMUM = {"CRITICAL": 0.0, "WARNING": 40.0,
                "WATCH": 60.0, "GOOD": 80.0}

# These authoritative states do not receive a numeric score. Returning an
# explicit reason avoids presenting missing/unreliable evidence as poor health.
NOT_COMPUTED_STATES = {
    "OFF": "PUMP_OFF",
    "DATA_ERROR": "DATA_ERROR",
    "DATA_QUALITY_ERROR": "DATA_QUALITY_ERROR",
    "WARMUP": "WARMUP",
    "MODEL_SCORE_UNAVAILABLE": "MODEL_SCORE_UNAVAILABLE",
    "BASELINE_PENDING": "BASELINE_PENDING",
    "STALE_DATA": "STALE_DATA",
    "CONFIGURATION_REQUIRED": "CONFIGURATION_REQUIRED",
    "PENDING_APPROVAL": "PENDING_APPROVAL",
}


def profile_health_baseline(profile: dict | None) -> dict | None:
    """Convert approved operating limits into health-scoring ranges."""
    if not profile or profile.get("approval_status") != "APPROVED":
        return None
    limits = profile.get("operating_limits") or {}
    ranges = {
        "current_avg": (limits.get("current_min_a"), limits.get("current_max_a")),
        "voltage_avg": (limits.get("voltage_min_v"), limits.get("voltage_max_v")),
        "total_kw": (limits.get("total_kw_min"), limits.get("total_kw_max")),
        "pf_avg": (limits.get("power_factor_min"), 1.0),
        "current_imbalance_pct": (0.0, limits.get("current_imbalance_max_percent")),
        "voltage_imbalance_pct": (0.0, limits.get("voltage_imbalance_max_percent")),
    }
    if limits.get("flowrate_min") is not None:
        ranges["flowrate"] = (limits["flowrate_min"], float("inf"))
    stats = {}
    for field, (low, high) in ranges.items():
        try:
            lower, upper = float(low), float(high)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(lower) or upper <= lower:
            continue
        width = upper - lower if math.isfinite(upper) else max(lower, 1.0)
        stats[field] = {"expected_low": lower, "expected_high": upper, "scale": max(width / 3, 0.01), "available": True}
    return {"stats": stats, "source": "ENGINEERING_PROFILE", "baseline_quality": "strong", "approved_healthy": True}

# Channel tuple: (reading/baseline field, penalized direction, display label).
# BOTH penalizes values below or above the expected range; LOW and HIGH only
# penalize the named side. OPERATING is event-derived, so it has no raw sensor.
GROUP_CHANNELS = {
    "ELECTRICAL": (
        ("current_imbalance_pct", "HIGH", "Current imbalance"),
        ("voltage_imbalance_pct", "HIGH", "Voltage imbalance"),
    ),
    "LOAD": (
        ("flowrate", "BOTH", "Flow"),
        ("total_kw", "BOTH", "Power"),
        ("current_avg", "BOTH", "Average current"),
    ),
    "POWER_QUALITY": (
        ("voltage_avg", "BOTH", "Average voltage"),
        ("pf_avg", "LOW", "Average power factor"),
    ),
    "OPERATING": (),
}

# Map each named fault to the group whose score must be severity-clamped.
FAULT_GROUPS = {
    "PHASE_LOSS": "ELECTRICAL",
    "VOLTAGE_PHASE_LOSS": "ELECTRICAL",
    "PHASE_CURRENT_IMBALANCE": "ELECTRICAL",
    "VOLTAGE_IMBALANCE": "ELECTRICAL",
    "DRY_RUN": "LOAD",
    "FLOW_RESTRICTION": "LOAD",
    "MOTOR_OVERLOAD": "LOAD",
    "UNDER_VOLTAGE": "POWER_QUALITY",
    "OVER_VOLTAGE": "POWER_QUALITY",
    "LOW_PF": "POWER_QUALITY",
    "SHORT_CYCLING": "OPERATING",
}

# Units used only in human-readable contributor evidence.
FIELD_UNITS = {
    "current_imbalance_pct": "%",
    "voltage_imbalance_pct": "%",
    "flowrate": "m3/h",
    "total_kw": "kW",
    "current_avg": "A",
    "voltage_avg": "V",
    "pf_avg": "",
}


def _empty_groups() -> dict[str, dict[str, Any]]:
    """Create the API structure used before group evidence is evaluated."""
    return {
        name: {"score": None, "weight_used": 0.0, "available": False}
        for name in GROUP_WEIGHTS
    }


def _unavailable_health(reason: str) -> dict[str, Any]:
    """Return an explicit not-computed result instead of a misleading zero."""
    return {
        "score": None,
        "band": None,
        "confidence": "UNAVAILABLE",
        "reason": reason,
        "cap_applied": None,
        "groups": _empty_groups(),
        "contributors": [],
        "model_flagged": False,
    }


def health_band(score: float) -> str:
    """Map a numeric score to the first inclusive boundary it satisfies."""
    for boundary, name in BAND_BOUNDARIES:
        if score >= boundary:
            return name
    return "CRITICAL"


def _interpolate_subscore(excess: float) -> float:
    """Convert normalized boundary excess to a 0-100 channel score.

    For adjacent points (x1, y1) and (x2, y2), the formula is:
    score = y1 + ((excess - x1) / (x2 - x1)) * (y2 - y1)
    """
    value = max(0.0, float(excess))
    for index in range(1, len(CHANNEL_BREAKPOINTS)):
        left_x, left_y = CHANNEL_BREAKPOINTS[index - 1]
        right_x, right_y = CHANNEL_BREAKPOINTS[index]
        if value <= right_x:
            position = (value - left_x) / (right_x - left_x)
            return left_y + position * (right_y - left_y)
    return CHANNEL_BREAKPOINTS[-1][1]


def _number(value: Any) -> float | None:
    """Return a finite float, or None for missing/invalid numeric input."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _channel_subscore(reading: Any, stats: dict, field: str,
                      direction_mode: str) -> tuple[float, str | None, float, float] | None:
    """Score one channel against its pump-specific expected range.

    LOW excess  = (expected_low - reading) / baseline_scale
    HIGH excess = (reading - expected_high) / baseline_scale
    Values inside the allowed direction/range score 100. Missing baselines,
    readings, or non-positive scales make the channel unavailable.
    """
    spec = available_stat(stats, field, ("scale",))
    value = _number(reading.get(field)) if reading is not None else None
    if spec is None or value is None:
        return None
    scale = _number(spec.get("scale"))
    if scale is None or scale <= 0:
        return None
    try:
        low, high = expected_range(spec)
    except (KeyError, TypeError, ValueError):
        return None

    direction = None
    bound = low
    if value < low and direction_mode in {"BOTH", "LOW"}:
        direction = "LOW"
        bound = low
        excess = (low - value) / scale
    elif value > high and direction_mode in {"BOTH", "HIGH"}:
        direction = "HIGH"
        bound = high
        excess = (value - high) / scale
    else:
        return 100.0, None, value, bound
    return _interpolate_subscore(excess), direction, value, bound


def _format_number(value: float) -> str:
    """Format evidence values with compact, magnitude-aware precision."""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _field_evidence(field: str, direction: str,
                    baseline_deviations: list[dict],
                    early_warnings: list[dict]) -> str | None:
    """Reuse detector evidence text that matches a field and direction."""
    prefix = field.upper()
    for item in [*baseline_deviations, *early_warnings]:
        code = str(item.get("code") or item.get("field") or "").upper()
        if code == field.upper() or code.startswith(f"{prefix}_"):
            return item.get("trend_evidence") or item.get("evidence")
        item_direction = str(item.get("direction") or "").upper()
        if item.get("field") == field and item_direction == direction:
            return item.get("trend_evidence") or item.get("evidence")
    return None


def _channel_text(label: str, field: str, direction: str,
                  value: float, bound: float) -> str:
    """Build fallback evidence when the detector supplied no matching text."""
    unit = FIELD_UNITS.get(field, "")
    suffix = f" {unit}" if unit else ""
    return (
        f"{label} {direction.lower()} "
        f"({_format_number(value)}{suffix} vs usual {_format_number(bound)}{suffix})"
    )


def _fault_tier(code: str) -> str:
    """Return CRITICAL for listed critical codes; otherwise default to MAJOR."""
    if code in CRITICAL_FAULTS:
        return "CRITICAL"
    return "MAJOR"


def _confidence(status: str, baseline: dict) -> str:
    """Describe evidence confidence without modifying the numeric score.

    Confidence is LIMITED on model errors, LOW when the baseline is not strong
    or an unapproved provisional baseline has fewer than 5,000 rows, and NORMAL
    otherwise.
    """
    if status == "MODEL_ERROR":
        return "LIMITED"
    if baseline.get("source") == "ENGINEERING_PROFILE":
        return "RULE_BASED"
    quality = str(baseline.get("baseline_quality") or "").lower()
    row_count = int(baseline.get("row_count") or 0)
    provisional_too_small = (
        not bool(baseline.get("approved_healthy")) and row_count < 5_000
    )
    if quality != "strong" or provisional_too_small:
        return "LOW"
    return "NORMAL"


def _cap_for_evidence(fault_codes: list[str], early_warnings: list[dict],
                      model_evidence: dict | None) -> tuple[float | None, str | None]:
    """Select the strictest applicable final-score cap and its explanation.

    Confirmed faults take precedence. Early-warning and model-only caps apply
    only when no named fault exists. min(candidates) selects the lowest cap.
    """
    candidates: list[tuple[float, str]] = []
    critical = sorted(code for code in fault_codes if code in CRITICAL_FAULTS)
    major = sorted(code for code in fault_codes if code not in CRITICAL_FAULTS)
    if critical:
        candidates.append((CRITICAL_FAULT_CAP,
                           f"CRITICAL_FAULT:{','.join(critical)}"))
    if major:
        candidates.append((MAJOR_FAULT_CAP, f"MAJOR_FAULT:{','.join(major)}"))
    if not fault_codes and early_warnings:
        candidates.append((EARLY_WARNING_CAP, "EARLY_WARNING"))
    if not fault_codes and model_evidence:
        candidates.append((MODEL_EVIDENCE_CAP, "MODEL_EVIDENCE"))
    return min(candidates, default=(None, None), key=lambda item: item[0] or 101)


def _base_assessment(
    session_reading: Any,
    baseline: dict | None,
    detected_faults: list[dict] | None,
    baseline_deviations: list[dict] | None,
    early_warnings: list[dict] | None,
    short_cycling: dict | None,
    model_evidence: dict | None,
    status: str,
    data_quality: dict | None,
) -> tuple[
    dict[str, Any],
    float | None,
    float | None,
    str | None,
    float | None,
]:
    """Build the unsmoothed health assessment and smoothing inputs.

    Returns (health, uncapped_score, cap, cap_name, floor). Keeping the score
    before cap/floor allows temporal smoothing to happen before current
    evidence constraints are reapplied.
    """
    normalized_status = str(status or "").upper()
    quality_status = str((data_quality or {}).get("status") or "").upper()
    if quality_status == "FAIL":
        return _unavailable_health("DATA_QUALITY_ERROR"), None, None, None, None
    if normalized_status in NOT_COMPUTED_STATES:
        reason = NOT_COMPUTED_STATES[normalized_status]
        return _unavailable_health(reason), None, None, None, None
    if baseline is None:
        return _unavailable_health("BASELINE_UNAVAILABLE"), None, None, None, None
    if session_reading is None:
        return _unavailable_health("SESSION_READING_UNAVAILABLE"), None, None, None, None

    faults = detected_faults or []
    deviations = baseline_deviations or []
    warnings = early_warnings or []
    short = short_cycling or {}
    fault_codes = [
        str(item.get("code") or "UNKNOWN").upper()
        for item in faults
    ]
    stats = baseline.get("stats") or {}
    groups = _empty_groups()
    contributors: list[dict[str, Any]] = []
    fault_contributors: list[tuple[int, dict[str, Any]]] = []

    # Convert every usable raw channel into a baseline-relative 0-100 score.
    group_inputs: dict[str, list[float]] = {name: [] for name in GROUP_WEIGHTS}
    for group, channels in GROUP_CHANNELS.items():
        for field, direction_mode, label in channels:
            result = _channel_subscore(session_reading, stats, field, direction_mode)
            if result is None:
                continue
            subscore, direction, value, bound = result
            group_inputs[group].append(subscore)
            points_lost = 100.0 - subscore
            if direction and points_lost > 0:
                text = _field_evidence(field, direction, deviations, warnings)
                contributors.append({
                    "name": label,
                    "group": group,
                    "points_lost": round(points_lost, 1),
                    "text": text or _channel_text(
                        label, field, direction, value, bound
                    ),
                })

    # Short-cycling score is based on excess starts rather than a sensor range:
    # score = max(0, 100 * (2 * allowed_starts - actual_starts) / allowed_starts)
    # It stays 100 up to the allowed count and reaches 0 at twice that count.
    if short.get("evaluated"):
        starts = max(0, int(short.get("starts_in_window") or 0))
        allowed = max(1, int(short.get("allowed_starts") or 4))
        if starts <= allowed:
            cycling_score = 100.0
        else:
            cycling_score = max(0.0, 100.0 * (2 * allowed - starts) / allowed)
        group_inputs["OPERATING"].append(cycling_score)
        if cycling_score < 100:
            contributors.append({
                "name": "Short cycling",
                "group": "OPERATING",
                "points_lost": round(100.0 - cycling_score, 1),
                "text": (
                    f"{starts} starts in the evaluation window "
                    f"(usual limit {allowed})"
                ),
            })

    # A group is intentionally represented by its weakest available channel so
    # a healthy companion sensor cannot hide a severely abnormal one.
    for group, values in group_inputs.items():
        if values:
            groups[group]["score"] = min(values)
            groups[group]["available"] = True

    # Confirmed fault evidence clamps its associated group to 10 (critical) or
    # 40 (major), regardless of a healthier raw channel score.
    for fault, code in zip(faults, fault_codes):
        group = FAULT_GROUPS.get(code)
        tier = _fault_tier(code)
        clamp = CRITICAL_GROUP_CLAMP if tier == "CRITICAL" else MAJOR_GROUP_CLAMP
        if group and groups[group]["available"]:
            groups[group]["score"] = min(groups[group]["score"], clamp)
        fault_contributors.append((
            0 if tier == "CRITICAL" else 1,
            {
                "name": fault.get("label") or code.replace("_", " ").title(),
                "group": group or "OTHER",
                "points_lost": 100.0 - clamp,
                "text": fault.get("evidence") or f"{code.replace('_', ' ').title()} detected",
            },
        ))

    available = [name for name, item in groups.items() if item["available"]]
    if not available:
        health = _unavailable_health("NO_HEALTH_INPUTS")
        health["model_flagged"] = bool(model_evidence)
        return health, None, None, None, None

    # Weighted score over available groups only:
    # weighted_score = sum(group_score * group_weight / available_weight_sum)
    total_weight = sum(GROUP_WEIGHTS[name] for name in available)
    weighted_score = 0.0
    for name, item in groups.items():
        if not item["available"]:
            continue
        normalized_weight = GROUP_WEIGHTS[name] / total_weight
        item["weight_used"] = round(normalized_weight, 4)
        item["score"] = round(float(item["score"]), 1)
        weighted_score += normalized_weight * item["score"]

    # Penalize simultaneous degradation across independent health dimensions.
    low_groups = sum(
        1 for item in groups.values()
        if item["available"] and item["score"] < LOW_GROUP_THRESHOLD
    )
    multi_issue_penalty = min(
        MULTI_ISSUE_PENALTY_MAX,
        MULTI_ISSUE_PENALTY_PER_EXTRA_GROUP * max(0, low_groups - 1),
    )
    # Raw score is always bounded to the public 0-100 contract.
    uncapped_score = max(0.0, min(100.0, weighted_score - multi_issue_penalty))
    cap_value, cap_name = _cap_for_evidence(fault_codes, warnings, model_evidence)
    floor_value = (
        EARLY_WARNING_FLOOR
        if not fault_codes and not model_evidence
        else None
    )
    # Evidence constraints are applied in order:
    # capped_score = min(raw_score, evidence_cap)
    # final_score  = max(capped_score, early_warning_floor), when applicable.
    capped_score = min(uncapped_score, cap_value) if cap_value is not None else uncapped_score
    final_score = max(capped_score, floor_value) if floor_value is not None else capped_score
    applied_cap = cap_name if cap_value is not None and final_score < uncapped_score else None

    fault_contributors.sort(
        key=lambda item: (item[0], -item[1]["points_lost"], item[1]["name"])
    )
    contributors.sort(key=lambda item: (-item["points_lost"], item["name"]))
    ordered_contributors = [item for _, item in fault_contributors] + contributors
    health = {
        "score": round(final_score, 1),
        "band": health_band(final_score),
        "confidence": _confidence(normalized_status, baseline),
        "reason": None,
        "cap_applied": applied_cap,
        "groups": groups,
        "contributors": ordered_contributors[:3],
        "model_flagged": bool(model_evidence),
    }
    if floor_value is not None and final_score > capped_score:
        health["floor_applied"] = True
    return health, uncapped_score, cap_value, cap_name, floor_value


def calculate_health(
    session_reading: Any,
    baseline: dict | None,
    detected_faults: list[dict] | None,
    baseline_deviations: list[dict] | None,
    early_warnings: list[dict] | None,
    short_cycling: dict | None,
    model_evidence: dict | None,
    status: str,
    data_quality: dict | None,
) -> dict[str, Any]:
    """Return an unsmoothed 0-100 health projection for one evaluated reading."""
    health, _, _, _, _ = _base_assessment(
        session_reading, baseline, detected_faults, baseline_deviations,
        early_warnings, short_cycling, model_evidence, status, data_quality,
    )
    return health


class HealthSmoother:
    """Stabilize health without hiding immediate deterioration.

    Each pump owns an independent fixed-length score window. Repeated refreshes
    of the same timestamp return the cached result and therefore cannot bias the
    median. Caps/floors are applied after the median so current confirmed
    evidence remains authoritative.
    """

    def __init__(self, window: int = HEALTH_SMOOTHING_WINDOW):
        """Initialize independent per-pump smoothing and hysteresis state."""
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self._scores: dict[str, deque[float]] = {}
        self._last_time: dict[str, datetime] = {}
        self._bands: dict[str, str] = {}
        self._last_result: dict[str, tuple[float, str, float]] = {}

    def reset(self, pump_id: str) -> None:
        """Discard all temporal health state for one pump."""
        self._scores.pop(pump_id, None)
        self._last_time.pop(pump_id, None)
        self._bands.pop(pump_id, None)
        self._last_result.pop(pump_id, None)

    @staticmethod
    def _timestamp(value: str | datetime) -> datetime:
        """Normalize ISO strings or datetimes to timezone-aware UTC."""
        stamp = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str) else value
        )
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)

    def _hysteretic_band(self, pump_id: str, score: float) -> str:
        """Apply immediate worsening and guarded recovery between bands.

        To recover into a better band:
        required_score = next_band_minimum + BAND_RECOVERY_HYSTERESIS
        Multiple bands may be recovered in one update only when every crossed
        threshold is satisfied.
        """
        candidate = health_band(score)
        previous = self._bands.get(pump_id)
        if previous is None:
            return candidate
        previous_index = BAND_ORDER.index(previous)
        candidate_index = BAND_ORDER.index(candidate)
        if candidate_index <= previous_index:
            return candidate
        improved = previous_index
        while improved < candidate_index:
            next_band = BAND_ORDER[improved + 1]
            threshold = BAND_MINIMUM[next_band] + BAND_RECOVERY_HYSTERESIS
            if score < threshold:
                break
            improved += 1
        return BAND_ORDER[improved]

    def update(self, pump_id: str, timestamp: str | datetime | None,
               score: float | None, cap: float | None = None,
               floor: float | None = None,
               ) -> tuple[float | None, str | None, float | None]:
        """Add one distinct score and return (final score, band, raw median).

        smoothed_uncapped = median(last N raw scores)
        final = clamp(max(min(smoothed_uncapped, cap), floor), 0, 100)
        Missing/invalid input resets pump state; old/duplicate timestamps reuse
        the cached result.
        """
        if timestamp is None or score is None:
            self.reset(pump_id)
            return None, None, None
        try:
            stamp = self._timestamp(timestamp)
            value = float(score)
            if not math.isfinite(value):
                raise ValueError("non-finite score")
        except (TypeError, ValueError, AttributeError):
            self.reset(pump_id)
            return None, None, None

        last = self._last_time.get(pump_id)
        if last is not None and stamp <= last:
            cached = self._last_result.get(pump_id)
            return cached if cached is not None else (None, None, None)

        values = self._scores.setdefault(pump_id, deque(maxlen=self.window))
        values.append(max(0.0, min(100.0, value)))
        uncapped = float(median(values))
        final = min(uncapped, cap) if cap is not None else uncapped
        final = max(final, floor) if floor is not None else final
        final = max(0.0, min(100.0, final))
        band = self._hysteretic_band(pump_id, final)
        self._last_time[pump_id] = stamp
        self._bands[pump_id] = band
        result = (round(final, 1), band, uncapped)
        self._last_result[pump_id] = result
        return result


def calculate_smoothed_health(
    smoother: HealthSmoother,
    pump_id: str,
    timestamp: str | datetime | None,
    session_reading: Any,
    baseline: dict | None,
    detected_faults: list[dict] | None,
    baseline_deviations: list[dict] | None,
    early_warnings: list[dict] | None,
    short_cycling: dict | None,
    model_evidence: dict | None,
    status: str,
    data_quality: dict | None,
) -> dict[str, Any]:
    """Calculate health, median-smooth the raw score, then reapply evidence.

    This is the stateful public path used when UI stability is required. It
    preserves the contributor/group explanation from the current assessment
    while replacing its score and band with the timestamp-safe smoothed result.
    """
    health, uncapped, cap_value, cap_name, floor_value = _base_assessment(
        session_reading, baseline, detected_faults, baseline_deviations,
        early_warnings, short_cycling, model_evidence, status, data_quality,
    )
    if health["score"] is None:
        smoother.reset(pump_id)
        return health
    score, band, smoothed_uncapped = smoother.update(
        pump_id, timestamp, uncapped, cap_value, floor_value
    )
    if score is None:
        health.update({
            "score": None,
            "band": None,
            "reason": "INVALID_HEALTH_TIMESTAMP",
            "cap_applied": None,
        })
        return health
    health["score"] = score
    health["band"] = band
    health["cap_applied"] = (
        cap_name
        if cap_value is not None and smoothed_uncapped is not None
        and smoothed_uncapped > cap_value
        else None
    )
    health.pop("floor_applied", None)
    if floor_value is not None and smoothed_uncapped is not None:
        smoothed_capped = (
            min(smoothed_uncapped, cap_value)
            if cap_value is not None else smoothed_uncapped
        )
        if smoothed_capped < floor_value:
            health["floor_applied"] = True
    return health
