"""Engineering-limit checks and forecast risk scoring for energy-meter telemetry."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PHASES = ("r", "y", "b")
FORECAST_FIELDS = tuple(
    [f"current_{phase}" for phase in PHASES]
    + [f"voltage_{phase}" for phase in PHASES]
    + [f"pf_{phase}" for phase in PHASES]
    + ["total_kw"]
)


def validate_profile(profile: dict[str, Any]) -> dict:
    """Validate engineer-approved motor details and operating limits."""
    required = ("rated_power_kw", "rated_voltage_v", "rated_current_a", "rated_frequency_hz", "phase_count", "operating_limits")
    missing = [key for key in required if profile.get(key) is None]
    if missing:
        raise ValueError("Missing required profile fields: " + ", ".join(missing))
    if int(profile["phase_count"]) != 3:
        raise ValueError("Only three-phase pumps are supported by this monitoring profile.")
    for key in ("rated_power_kw", "rated_voltage_v", "rated_current_a", "rated_frequency_hz"):
        if not float(profile[key]) > 0:
            raise ValueError(f"{key} must be greater than zero.")
    limits = profile["operating_limits"]
    for key in ("current_min_a", "voltage_min_v", "total_kw_min"):
        if key in limits and float(limits[key]) < 0:
            raise ValueError(f"{key} cannot be negative.")
    for low, high in (("current_min_a", "current_max_a"), ("voltage_min_v", "voltage_max_v"), ("total_kw_min", "total_kw_max")):
        if low not in limits or high not in limits or float(limits[low]) >= float(limits[high]):
            raise ValueError(f"{low} must be lower than {high}.")
    if not 0 < float(limits.get("power_factor_min", 0)) <= 1:
        raise ValueError("power_factor_min must be between 0 and 1.")
    for key in ("current_imbalance_max_percent", "voltage_imbalance_max_percent"):
        if float(limits.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive.")
    if limits.get("flowrate_min") is None:
        raise ValueError("flowrate_min (minimum delivery flow) is required.")
    if not float(limits["flowrate_min"]) > 0:
        raise ValueError("flowrate_min must be greater than zero.")
    return profile


def profile_limit_warnings(
    profile: dict[str, Any],
    *,
    voltage_tolerance_percent: float = 25.0,
    max_to_rated_ratio: float = 1.5,
    min_max_to_rated_ratio: float = 0.5,
) -> list[dict]:
    """Flag limits that look inconsistent with this pump's own nameplate.

    Every product has different limits, so nothing here is rejected: these are
    prompts for the engineer to confirm, measured relative to the pump's rated
    voltage, current and power rather than against fixed numbers.
    """
    limits = profile["operating_limits"]
    warnings: list[dict] = []

    def warn(code: str, fields: list[str], message: str) -> None:
        warnings.append({"code": code, "fields": fields, "message": message})

    rated_voltage = float(profile["rated_voltage_v"])
    voltage_min, voltage_max = float(limits["voltage_min_v"]), float(limits["voltage_max_v"])
    tolerance = voltage_tolerance_percent / 100
    if not voltage_min <= rated_voltage <= voltage_max:
        warn("VOLTAGE_RANGE_EXCLUDES_RATED", ["voltage_min_v", "voltage_max_v"],
             f"Voltage range {voltage_min:g}-{voltage_max:g} V does not include the rated {rated_voltage:g} V.")
    elif voltage_min < rated_voltage * (1 - tolerance) or voltage_max > rated_voltage * (1 + tolerance):
        warn("VOLTAGE_RANGE_FAR_FROM_RATED", ["voltage_min_v", "voltage_max_v"],
             f"Voltage range {voltage_min:g}-{voltage_max:g} V goes beyond ±{voltage_tolerance_percent:g}% of the rated "
             f"{rated_voltage:g} V ({rated_voltage * (1 - tolerance):g}-{rated_voltage * (1 + tolerance):g} V).")

    for field, rated_key, unit, label in (("current_max_a", "rated_current_a", "A", "Current maximum"),
                                          ("total_kw_max", "rated_power_kw", "kW", "Maximum operating load")):
        value, rated = float(limits[field]), float(profile[rated_key])
        if value > rated * max_to_rated_ratio:
            warn(f"{field.upper()}_ABOVE_RATED", [field],
                 f"{label} {value:g} {unit} is more than {max_to_rated_ratio:g}x the rated {rated:g} {unit}.")
        elif value < rated * min_max_to_rated_ratio:
            warn(f"{field.upper()}_BELOW_RATED", [field],
                 f"{label} {value:g} {unit} is less than {min_max_to_rated_ratio:g}x the rated {rated:g} {unit}; "
                 "normal running would be flagged as a fault.")
    return warnings


def _imbalance(values: list[float]) -> float:
    """Return maximum phase deviation from the average as a percentage."""
    average = float(np.mean(values))
    return 0.0 if average <= 0 else max(abs(value - average) for value in values) / average * 100.0


HYPHEN_COLUMN = "hyphen_fields"


def mark_hyphen_rows(frame: pd.DataFrame, fields) -> list[str]:
    """Record, per row, which sensor fields arrived as '-' before numeric coercion.

    Coercion turns '-' into a blank that looks like an ordinary missing value, so the
    data-quality firewall reads this column instead. Existing marks are kept.
    Returns the fields that contained any '-'.
    """
    existing = frame[HYPHEN_COLUMN] if HYPHEN_COLUMN in frame.columns else None
    marks = (existing.fillna("").astype(str) if existing is not None else pd.Series("", index=frame.index, dtype=object)).copy()
    found = []
    for field in fields:
        if field not in frame.columns or frame[field].dtype != object:
            continue
        mask = frame[field].astype(str).str.strip().eq("-")
        if mask.any():
            found.append(field)
            marks.loc[mask] = marks.loc[mask].map(
                lambda value, name=field: value if name in value.split(",") else (f"{value},{name}" if value else name)
            )
    frame[HYPHEN_COLUMN] = marks
    return found


NONFINITE_COLUMN = "nonfinite_fields"


def mark_nonfinite_rows(frame: pd.DataFrame, fields) -> list[str]:
    """Blank +/-inf sensor values and record, per row, which fields held them.

    A meter can send text such as "inf" that numeric coercion turns into an
    infinite number. Blanking it keeps infinities out of charts, averages and
    JSON, while the marker lets the data-quality firewall reject that reading.
    Existing marks are kept. Returns the fields that contained an infinity.
    """
    existing = frame[NONFINITE_COLUMN] if NONFINITE_COLUMN in frame.columns else None
    marks = (existing.fillna("").astype(str) if existing is not None else pd.Series("", index=frame.index, dtype=object)).copy()
    found = []
    for field in fields:
        if field not in frame.columns:
            continue
        values = pd.to_numeric(frame[field], errors="coerce")
        mask = values.isin([np.inf, -np.inf])
        if not mask.any():
            continue
        found.append(field)
        frame[field] = values.mask(mask)
        marks.loc[mask] = marks.loc[mask].map(
            lambda value, name=field: value if name in value.split(",") else (f"{value},{name}" if value else name)
        )
    if found or existing is not None:
        frame[NONFINITE_COLUMN] = marks
    return found


def enriched_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce forecast fields and add phase-imbalance features."""
    work = frame.copy()
    if any(field in work and work[field].dtype == object for field in FORECAST_FIELDS):
        mark_hyphen_rows(work, FORECAST_FIELDS)
    mark_nonfinite_rows(work, [*FORECAST_FIELDS, "flowrate"])
    for field in FORECAST_FIELDS:
        if field in work:
            work[field] = pd.to_numeric(work[field], errors="coerce")
    for prefix, target in (("current", "current_imbalance_pct"), ("voltage", "voltage_imbalance_pct")):
        cols = [f"{prefix}_{phase}" for phase in PHASES]
        if set(cols).issubset(work.columns):
            work[target] = work[cols].apply(
                lambda row: _imbalance([float(value) for value in row]) if row.notna().all() else np.nan,
                axis=1,
            )
    return work


def current_faults(row: pd.Series, profile: dict) -> list[dict]:
    """Flag faults in the latest reading against approved operating limits."""
    limits = profile["operating_limits"]
    faults: list[dict] = []
    def add(code: str, label: str, evidence: str) -> None:
        """Append one consistently structured live engineering fault."""
        faults.append({"code": code, "label": label, "detection_source": "ENGINEERING_PROFILE", "evidence": evidence})

    def phase_values(prefix: str) -> list[float]:
        return [float(pd.to_numeric(row.get(f"{prefix}_{phase}"), errors="coerce")) for phase in PHASES]

    def finite(values: list[float]) -> list[float]:
        # Python min()/max() give order-dependent answers when a value is NaN, so a
        # blank phase must be dropped before ranking the remaining phases.
        return [value for value in values if np.isfinite(value)]

    currents = phase_values("current")
    voltages = phase_values("voltage")
    pfs = phase_values("pf")
    known_currents, known_voltages, known_pfs = finite(currents), finite(voltages), finite(pfs)
    power = float(pd.to_numeric(row.get("total_kw"), errors="coerce"))
    flow = pd.to_numeric(pd.Series([row.get("flowrate")]), errors="coerce").iloc[0]
    current_imbalance = _imbalance(currents)
    voltage_imbalance = _imbalance(voltages)
    # A lost phase leaves one current near zero while the others still carry clear
    # current. "Clear current" is 2 A (or half the approved minimum for small pumps,
    # never below 1 A) so a lightly loaded pump is covered but a stopping pump is not.
    loaded_phase_current = min(2.0, max(1.0, float(limits["current_min_a"]) * 0.5))
    if known_currents and min(known_currents) <= 0.5 and max(known_currents) > loaded_phase_current:
        add("PHASE_LOSS", "Phase Loss", "One phase current is near zero while the other phases are still carrying current.")
    approved_current_max = float(limits["current_max_a"])
    rated_current = float(profile["rated_current_a"])
    overload_limit = min(approved_current_max, rated_current)
    if known_currents and max(known_currents) > overload_limit:
        limit_source = "approved operating maximum" if approved_current_max <= rated_current else "rated current"
        add(
            "MOTOR_OVERLOAD",
            "Motor Overload",
            f"Phase current {max(known_currents):.2f} A exceeds the {limit_source} {overload_limit:.2f} A.",
        )
    if current_imbalance > float(limits["current_imbalance_max_percent"]):
        add("PHASE_CURRENT_IMBALANCE", "Current Imbalance", f"Current imbalance is {current_imbalance:.1f}%.")
    phase_voltage_loss_threshold = max(5.0, float(limits["voltage_min_v"]) * .10)
    if known_voltages and min(known_voltages) <= phase_voltage_loss_threshold and max(known_voltages) >= float(limits["voltage_min_v"]):
        add("VOLTAGE_PHASE_LOSS", "Voltage Phase Loss", "One phase voltage is near zero while the other phases remain energized.")
    if known_voltages and min(known_voltages) < float(limits["voltage_min_v"]):
        add("UNDER_VOLTAGE", "Under Voltage", f"Lowest phase voltage is {min(known_voltages):.1f} V.")
    if known_voltages and max(known_voltages) > float(limits["voltage_max_v"]):
        add("OVER_VOLTAGE", "Over Voltage", f"Highest phase voltage is {max(known_voltages):.1f} V.")
    if voltage_imbalance > float(limits["voltage_imbalance_max_percent"]):
        add("VOLTAGE_IMBALANCE", "Voltage Imbalance", f"Voltage imbalance is {voltage_imbalance:.1f}%.")
    if known_pfs and np.mean(known_pfs) < float(limits["power_factor_min"]):
        add("LOW_PF", "Low Power Factor", f"Average power factor is {np.mean(known_pfs):.3f}.")
    if power > float(limits["total_kw_max"]):
        add("HIGH_POWER", "High Power", f"Total power is {power:.2f} kW.")
    flow_minimum = limits.get("flowrate_min")
    has_loaded_draw = bool(known_currents) and max(known_currents) >= float(limits["current_min_a"]) and power >= float(limits["total_kw_min"])
    if flow_minimum is not None and np.isfinite(flow) and has_loaded_draw:
        minimum_flow = float(flow_minimum)
        dry_run_threshold = min(1.0, minimum_flow * .10)
        if flow <= dry_run_threshold:
            add("DRY_RUN", "Possible Dry Run", f"The pump is drawing power but water flow is only {flow:.1f} m³/h (dry running is {dry_run_threshold:.1f} m³/h or less).")
        elif flow < minimum_flow:
            add("FLOW_RESTRICTION", "Possible Flow Restriction", f"The pump is drawing power but water flow is {flow:.1f} m³/h, below the approved minimum of {minimum_flow:.1f} m³/h.")
    return faults


def _severity(value: float, normal: float, critical: float, *, high: bool) -> float:
    """Scale a limit deviation into a bounded zero-to-one severity."""
    if high:
        return float(np.clip((value - normal) / max(critical - normal, 1e-6), 0, 1))
    return float(np.clip((normal - value) / max(normal - critical, 1e-6), 0, 1))


def _interpolate_short_internal_gaps(values: list[float], *, max_gap: int = 2) -> tuple[list[float], int]:
    """Fill only bounded internal non-finite runs no longer than max_gap."""
    numeric = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype=float)
    filled = numeric.copy()
    imputed = 0
    index = 0
    while index < len(filled):
        if np.isfinite(filled[index]):
            index += 1
            continue
        start = index
        while index < len(filled) and not np.isfinite(filled[index]):
            index += 1
        end = index
        gap = end - start
        if start == 0 or end == len(filled) or gap > max_gap:
            continue
        left, right = filled[start - 1], filled[end]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        step = (right - left) / (gap + 1)
        for offset in range(gap):
            filled[start + offset] = left + step * (offset + 1)
        imputed += gap
    return filled.tolist(), imputed


def _quality_checked_prediction(
    prediction: dict[str, dict[str, list[float]]],
    *,
    minimum_valid_fraction: float = .70,
) -> tuple[dict[str, dict[str, list[float]]] | None, dict[str, Any]]:
    """Return a sanitized copy plus quality metadata for risk calculation."""
    sanitized: dict[str, dict[str, list[float]]] = {}
    total_points = 0
    valid_points = 0
    imputed_points = 0
    unavailable_series: list[str] = []
    expected_length: int | None = None
    for field in FORECAST_FIELDS:
        sanitized[field] = {}
        source = prediction.get(field, {})
        for quantile in ("q05", "q50", "q95"):
            values = source.get(quantile)
            if not isinstance(values, list) or not values:
                unavailable_series.append(f"{field}.{quantile}")
                continue
            if expected_length is None:
                expected_length = len(values)
            if len(values) != expected_length:
                unavailable_series.append(f"{field}.{quantile}")
                continue
            numeric = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype=float)
            total_points += len(numeric)
            valid_points += int(np.isfinite(numeric).sum())
            filled, count = _interpolate_short_internal_gaps(values)
            sanitized[field][quantile] = filled
            imputed_points += count
    valid_fraction = valid_points / total_points if total_points else 0.0
    usable = not unavailable_series and valid_fraction >= minimum_valid_fraction
    confidence = "HIGH" if valid_fraction >= .90 else "LOW" if usable else "UNAVAILABLE"
    quality = {
        "valid_fraction": round(valid_fraction, 4),
        "imputed_points": imputed_points,
        "confidence": confidence,
        "minimum_valid_fraction": minimum_valid_fraction,
        "unavailable_series": unavailable_series,
    }
    return (sanitized if usable else None), quality


def forecast_risk(prediction: dict[str, dict[str, list[float]]], profile: dict, interval_minutes: int = 5) -> dict:
    """Turn predicted electrical quantiles into advisory limit-crossing signals.

    This score is not a calibrated probability of pump failure.
    """
    prediction, quality = _quality_checked_prediction(prediction)
    if prediction is None:
        return {
            "state": "INSUFFICIENT_FORECAST_QUALITY",
            "risk_score": None,
            "attention_level": "DATA_UNAVAILABLE",
            "signals": [],
            "forecast_quality": quality,
        }
    limits = profile["operating_limits"]
    scores: list[tuple[str, float, int | None, str]] = []
    current_warning = float(limits["current_max_a"])
    current_critical = min(float(profile["rated_current_a"]), current_warning + (current_warning - float(limits["current_min_a"])))
    for phase in PHASES:
        values = prediction[f"current_{phase}"]["q95"]
        finite_values = [value for value in values if np.isfinite(value)]
        score = max((_severity(value, current_warning, current_critical, high=True) for value in finite_values), default=0.0)
        hit = next((index for index, value in enumerate(values) if value > current_warning), None)
        if score or hit is not None:
            scores.append(("RISING_LOAD_TREND", score, hit, "Motor current is forecast to come close to its safe limit."))
    voltage_low = [prediction[f"voltage_{phase}"]["q05"] for phase in PHASES]
    low_by_step = [min(series[index] for series in voltage_low) for index in range(len(voltage_low[0]))]
    low_score = max((_severity(value, float(limits["voltage_min_v"]), float(limits["voltage_min_v"]) * .9, high=False) for value in low_by_step if np.isfinite(value)), default=0.0)
    low_hit = next((index for index, value in enumerate(low_by_step) if np.isfinite(value) and value < float(limits["voltage_min_v"])), None)
    if low_score:
        scores.append(("VOLTAGE_STABILITY_DEGRADING", low_score, low_hit, "Voltage is forecast to fall below its safe range."))
    voltage_high = [prediction[f"voltage_{phase}"]["q95"] for phase in PHASES]
    high_by_step = [max(series[index] for series in voltage_high) for index in range(len(voltage_high[0]))]
    high_score = max((_severity(value, float(limits["voltage_max_v"]), float(limits["voltage_max_v"]) * 1.1, high=True) for value in high_by_step if np.isfinite(value)), default=0.0)
    high_hit = next((index for index, value in enumerate(high_by_step) if np.isfinite(value) and value > float(limits["voltage_max_v"])), None)
    if high_score:
        scores.append(("OVER_VOLTAGE_TREND", high_score, high_hit, "Voltage is forecast to rise above its safe range."))
    pf_series = [prediction[f"pf_{phase}"]["q05"] for phase in PHASES]
    pf_by_step = [min(series[index] for series in pf_series) for index in range(len(pf_series[0]))]
    pf_score = max((_severity(value, float(limits["power_factor_min"]), max(.01, float(limits["power_factor_min"]) - .15), high=False) for value in pf_by_step if np.isfinite(value)), default=0.0)
    pf_hit = next((index for index, value in enumerate(pf_by_step) if np.isfinite(value) and value < float(limits["power_factor_min"])), None)
    if pf_score:
        scores.append(("POWER_FACTOR_DEGRADING", pf_score, pf_hit, "Power factor is forecast to fall below its approved minimum."))
    median_currents = [prediction[f"current_{phase}"]["q50"] for phase in PHASES]
    imbalance: list[tuple[int, float]] = []
    for index in range(len(median_currents[0])):
        phase_values = [series[index] for series in median_currents]
        if all(np.isfinite(value) for value in phase_values):
            imbalance.append((index, _imbalance(phase_values)))
    imbalance_score = max((_severity(value, float(limits["current_imbalance_max_percent"]), float(limits["current_imbalance_max_percent"]) * 2, high=True) for _, value in imbalance), default=0.0)
    hit = next((index for index, value in imbalance if value > float(limits["current_imbalance_max_percent"])), None)
    if imbalance_score or hit is not None:
        scores.append(("PHASE_DIVERGENCE_TREND", imbalance_score, hit, "The three phase currents are forecast to become less balanced."))
    power = prediction["total_kw"]["q95"]
    power_warning = float(limits["total_kw_max"])
    power_critical = max(float(profile["rated_power_kw"]), power_warning + (power_warning - float(limits["total_kw_min"])))
    power_score = max((_severity(value, power_warning, power_critical, high=True) for value in power if np.isfinite(value)), default=0.0)
    power_hit = next((index for index, value in enumerate(power) if np.isfinite(value) and value > power_warning), None)
    if power_score:
        scores.append(("ENERGY_INTENSITY_DRIFT", power_score, power_hit, "Power is forecast to go above its safe range."))
    grouped: dict[str, tuple[float, int | None, str]] = {}
    for code, score, hit, message in scores:
        old = grouped.get(code)
        if old is None or score > old[0]:
            grouped[code] = (score, hit, message)
    weighted = {"RISING_LOAD_TREND": .25, "PHASE_DIVERGENCE_TREND": .25, "POWER_FACTOR_DEGRADING": .20, "VOLTAGE_STABILITY_DEGRADING": .15, "OVER_VOLTAGE_TREND": .15, "ENERGY_INTENSITY_DRIFT": .15}
    weighted_total = sum(weighted[code] * score for code, (score, _, _) in grouped.items())
    # A single severe forecast should not be hidden just because other channels
    # are normal; weights express contribution, not permission to ignore danger.
    strongest_signal = max((score for score, _, _ in grouped.values()), default=0.0)
    total = min(1.0, max(weighted_total, strongest_signal * .75))
    signals = []
    for code, (score, hit, message) in grouped.items():
        signals.append({"code": code, "severity": round(score, 3), "expected_in_minutes": None if hit is None else (hit + 1) * interval_minutes, "evidence": message})
    # Two operator levels only: the forecast either needs maintenance attention
    # (a limit is forecast to be crossed, or the combined risk is material) or it
    # does not. risk_score and each signal's timing still show how serious it is.
    crossing_forecast = any(signal["expected_in_minutes"] is not None for signal in signals)
    level = "MAINTENANCE_ATTENTION" if crossing_forecast or total >= .25 else "NORMAL"
    return {
        "state": "READY",
        "risk_score": round(total, 3),
        "attention_level": level,
        "signals": signals,
        "forecast_quality": quality,
    }
