"""Suggest a pump's nameplate values and operating limits from its running history.

Used by the admin "Calculate limits from data" button. The engineer reviews the values
before saving. The method matches how limits were set by hand for the first pumps:

* rated power: the next standard motor size so the pump normally runs below ~90% of it;
* rated current: the larger of that size's full-load current and the highest running
  phase current + 15%;
* voltage limits stay at +/-10% of the meter's voltage (never widened to hide a supply
  problem); minimum PF and current imbalance follow the pump's own normal operation;
* the result is replayed through the live rules to show which alarms the suggested
  limits would have raised over the same history.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

from backend.engine.electrical_profile import _imbalance, current_faults
from backend.engine.flow_suggestion import suggest_minimum_flow

STANDARD_MOTOR_KW = (0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.7, 5.5, 7.5, 9.3, 11, 15, 18.5, 22, 30, 37, 45,
                     55, 75, 90, 110, 132, 160, 200, 250)
CURRENTS = ["current_r", "current_y", "current_b"]
VOLTAGES = ["voltage_r", "voltage_y", "voltage_b"]
PFS = ["pf_r", "pf_y", "pf_b"]


def full_load_current(kw: float) -> float:
    """Full-load current of a 415 V three-phase motor (PF 0.85, efficiency 0.88)."""
    return kw * 1000 / (math.sqrt(3) * 415 * 0.85 * 0.88)


def suggest_limits(frame: pd.DataFrame, *, sample_minutes: int, days_requested: int, days_read: int,
                   days_failed: int, min_running_hours: float) -> dict:
    """Return suggested profile values, how they were found, warnings and expected alarms."""
    base = {"days_requested": days_requested, "days_read": days_read, "days_failed": days_failed,
            "sample_minutes": sample_minutes, "running_hours": 0.0, "readings": 0,
            "profile": None, "warnings": [], "expected_alarms": [], "basis": {}}
    if frame is None or frame.empty or not set(CURRENTS + ["total_kw"]).issubset(frame.columns):
        return {**base, "state": "NO_RUNNING_DATA",
                "message": f"No usable running data in the last {days_requested} days. Enter the nameplate values by hand."}
    work = frame.copy()
    # Running = drawing real power and current (status ON alone is not enough).
    drawing = (work["total_kw"] > 0.1) & (work[CURRENTS].max(axis=1) > 0.5)
    work = work.loc[drawing]
    hours = round(len(work) * sample_minutes / 60, 1)
    base.update(running_hours=hours, readings=int(len(work)))
    if hours < min_running_hours:
        return {**base, "state": "INSUFFICIENT_DATA",
                "message": f"Only {hours:g} running hours in the last {days_requested} days; at least {min_running_hours:g} are needed. Enter the nameplate values by hand."}

    kw = work["total_kw"]
    max_current = work[CURRENTS].max(axis=1)
    kw_median, kw_p99 = float(kw.median()), float(kw.quantile(.99))
    size = next((s for s in STANDARD_MOTOR_KW if s >= kw_median / 0.9 and s * 1.1 >= kw_p99), STANDARD_MOTOR_KW[-1])
    current_p99 = float(max_current.quantile(.99))
    rated_a = int(math.ceil(max(full_load_current(size), current_p99 * 1.15)))

    voltage_median = float(work[[c for c in VOLTAGES if c in work]].median().median())
    rated_v = 240 if voltage_median < 330 else 415
    warnings: list[str] = []
    if rated_v == 415:
        warnings.append("The meter reports line voltage (~415 V); limits use 415 V.")

    pf_values = work[[c for c in PFS if c in work]].mean(axis=1).dropna()
    # Lowest normal PF = bottom 1% of running readings. Keep the standard 0.8 when the pump
    # normally stays above 0.82; otherwise set the minimum 0.05 below its lowest normal PF.
    pf_p10 = float(pf_values.quantile(.01)) if len(pf_values) else float("nan")
    if np.isfinite(pf_p10) and float(pf_values.median()) < 0:
        warnings.append("Power factor comes as a negative number (meter CT reversed?). Low PF alarms will be wrong until the meter is fixed.")
        pf_min = 0.8
    elif not np.isfinite(pf_p10) or pf_p10 >= 0.82:
        pf_min = 0.8
    else:
        pf_min = max(0.3, math.floor((pf_p10 - 0.05) * 20) / 20)

    imbalance = work[CURRENTS].apply(lambda row: _imbalance(list(row)) if row.notna().all() else np.nan, axis=1)
    imbalance_p99 = float(imbalance.quantile(.99)) if imbalance.notna().any() else 0.0
    current_imbalance_max = 10 if imbalance_p99 <= 9 else int(min(20, math.ceil(imbalance_p99 + 3)))

    flows = work["flowrate"].dropna().tolist() if "flowrate" in work else []
    flow = suggest_minimum_flow(flows, window_minutes=sample_minutes, margin_percent=30,
                                minimum_running_hours=min(12, min_running_hours), lookback_days=days_requested)
    flow_min = flow["suggested_flowrate_min"] if flow.get("state") == "READY" else 1.0
    flow_note = flow.get("message") if flow.get("state") == "READY" else "No usable flow readings; minimum flow set to 1 m³/h."

    limits = {
        "current_min_a": round(rated_a * .5, 2), "current_max_a": rated_a,
        "voltage_min_v": round(rated_v * .9, 2), "voltage_max_v": round(rated_v * 1.1, 2),
        "power_factor_min": pf_min, "total_kw_min": round(size * .5, 2), "total_kw_max": round(size * 1.1, 2),
        "current_imbalance_max_percent": current_imbalance_max, "voltage_imbalance_max_percent": 3,
        "flowrate_min": flow_min,
    }
    profile = {"rated_power_kw": size, "rated_voltage_v": rated_v, "rated_current_a": rated_a,
               "rated_frequency_hz": 50, "phase_count": 3, "operating_limits": limits}

    voltages = work[[c for c in VOLTAGES if c in work]]
    v_low, v_high = float(voltages.min(axis=1).quantile(.01)), float(voltages.max(axis=1).quantile(.99))
    if v_low < limits["voltage_min_v"] or v_high > limits["voltage_max_v"]:
        warnings.append(f"Supply voltage went {v_low:.0f}–{v_high:.0f} V, outside {limits['voltage_min_v']:g}–{limits['voltage_max_v']:g} V. "
                        "Voltage limits are kept at ±10%; check the site supply.")
    voltage_imbalance = voltages.apply(lambda row: _imbalance(list(row)) if row.notna().all() else np.nan, axis=1)
    if voltage_imbalance.notna().any() and float(voltage_imbalance.quantile(.99)) > 3:
        warnings.append(f"Phase voltages differ by up to {float(voltage_imbalance.quantile(.99)):.0f}%; Voltage Imbalance alarms will appear.")
    lost_phase = ((work[CURRENTS].min(axis=1) <= 0.5) & (max_current > 2)).mean()
    if lost_phase > 0.05:
        warnings.append(f"One phase current is near zero {lost_phase:.0%} of the running time (phase loss or meter CT problem).")
    if "flowrate" in work and work["flowrate"].notna().any() and (work["flowrate"] <= 0).mean() > 0.10:
        warnings.append(f"Flow reads 0 for {(work['flowrate'] <= 0).mean():.0%} of the running time; Dry Run alarms will appear (check the flow meter).")
    if imbalance_p99 > 9:
        warnings.append(f"Phase currents normally differ by up to {imbalance_p99:.0f}%; current imbalance limit raised to {current_imbalance_max}%.")
    if hours < 72:
        warnings.append(f"Only {hours:g} running hours of data; check these values again after the pump has run more.")
    if days_failed:
        warnings.append(f"{days_failed} of {days_requested} days could not be read from the database and were skipped.")

    replay = {"rated_current_a": rated_a, "rated_power_kw": size, "operating_limits": limits}
    hits = Counter(fault["code"] for _, row in work.iterrows() for fault in current_faults(row, replay))
    expected = [{"code": code, "share_percent": round(count / len(work) * 100, 1)} for code, count in hits.most_common()]

    return {
        **base, "state": "READY", "profile": profile, "warnings": warnings, "expected_alarms": expected,
        "basis": {
            "running_kw_median": round(kw_median, 2), "running_kw_p99": round(kw_p99, 2),
            "running_current_p99": round(current_p99, 2), "full_load_current": round(full_load_current(size), 2),
            "power_factor_lowest_normal": None if not np.isfinite(pf_p10) else round(pf_p10, 3),
            "current_imbalance_p99": round(imbalance_p99, 1), "voltage_p1": round(v_low, 1), "voltage_p99": round(v_high, 1),
            "flow_note": flow_note,
        },
        "message": (f"Based on {hours:g} running hours over {days_read} of the last {days_requested} days. "
                    f"Runs at {kw_median:.2f} kW and up to {current_p99:.1f} A, so rated {size:g} kW / {rated_a} A."),
    }
