"""Suggest a pump's minimum delivery flow from its own running history.

The engineer presses "Check minimum flow from data" in the admin form. We take
the flow readings from the times the pump was running, remove unusual readings
(outliers), take the lowest remaining flow and allow a margin below it. The
engineer still reviews and saves the value.
"""
from __future__ import annotations

import numpy as np

# Readings below this share of the usual flow are always fault periods
# (dry running, meter dropouts), even when they are frequent.
FAULT_FLOW_SHARE = 0.10
# IQR rule: readings below Q1 - 1.5 x IQR (or above Q3 + 1.5 x IQR) are outliers.
IQR_FACTOR = 1.5
# A very steady pump has an IQR close to zero, which would mark normal meter
# noise as outliers. The spread used is never below this share of the usual flow.
MINIMUM_SPREAD_SHARE = 0.05
# The IQR rule cannot remove a problem that lasted over about a quarter of the
# running time (it becomes part of "normal"). If the lowest normal flow is still
# below this share of the usual flow, the engineer is warned to confirm it.
LOW_FLOW_WARNING_SHARE = 0.60


def suggest_minimum_flow(
    flows: list[float],
    *,
    window_minutes: int,
    margin_percent: float,
    minimum_running_hours: float,
    lookback_days: int,
    outlier_warning_percent: float = 20.0,
) -> dict:
    """Return the suggested minimum flow and how it was worked out."""
    values = np.asarray([value for value in flows if value is not None and np.isfinite(value) and value >= 0], dtype=float)
    hours = lambda count: round(count * window_minutes / 60, 1)  # noqa: E731 - small local unit helper
    running_hours = hours(len(values))
    result = {
        "lookback_days": lookback_days,
        "running_hours": running_hours,
        "minimum_running_hours": minimum_running_hours,
        "margin_percent": margin_percent,
        "typical_flow": None,
        "lowest_normal_flow": None,
        "outlier_hours": 0.0,
        "outlier_percent": 0.0,
        "warning": None,
        "suggested_flowrate_min": None,
    }
    if not len(values):
        return {**result, "state": "NO_RUNNING_DATA",
                "message": f"The pump has no running data in the last {lookback_days} days."}
    typical = float(np.median(values))
    result["typical_flow"] = round(typical, 1)
    if typical <= 0:
        return {**result, "state": "NO_FLOW_READINGS",
                "message": "The pump ran, but the flow meter showed no water. Check the flow meter, then enter the value by hand."}
    if running_hours < minimum_running_hours:
        return {**result, "state": "INSUFFICIENT_DATA",
                "message": f"Only {running_hours:g} running hours in the last {lookback_days} days; at least {minimum_running_hours:g} are needed."}

    # 1. Fault readings (dry run, meter dropouts) are never normal flow.
    working = values[values >= typical * FAULT_FLOW_SHARE]
    # 2. IQR rule on what is left.
    q1, q3 = np.percentile(working, [25, 75])
    spread = max(float(q3 - q1), typical * MINIMUM_SPREAD_SHARE)
    low_fence, high_fence = q1 - IQR_FACTOR * spread, q3 + IQR_FACTOR * spread
    normal = working[(working >= low_fence) & (working <= high_fence)]
    outliers = len(values) - len(normal)
    outlier_percent = round(outliers / len(values) * 100, 1)
    result.update(outlier_hours=hours(outliers), outlier_percent=outlier_percent)
    if not len(normal):
        return {**result, "state": "NO_FLOW_READINGS",
                "message": "No normal flow readings were left after removing unusual ones. Enter the value by hand."}

    # 3. Lowest normal flow, then the margin below it.
    lowest_normal = float(normal.min())
    suggested = round(lowest_normal * (1 - margin_percent / 100), 1)
    result["lowest_normal_flow"] = round(lowest_normal, 1)
    if suggested <= 0:
        return {**result, "state": "NO_FLOW_READINGS",
                "message": "The flow readings are too low to suggest a minimum. Enter the value by hand."}
    if outlier_percent > outlier_warning_percent:
        result["warning"] = (
            f"{outlier_percent:g}% of the running time had unusual flow readings. "
            "This data looks unusual; confirm the value with the site before using it."
        )
    elif lowest_normal < typical * LOW_FLOW_WARNING_SHARE:
        result["warning"] = (
            f"The flow often dropped to {lowest_normal:.1f} m³/h, well below the usual {typical:.1f} m³/h. "
            "The pump may have had a problem during this period; confirm the value with the site before using it."
        )
    return {
        **result,
        "state": "READY",
        "suggested_flowrate_min": suggested,
        "message": (
            f"Based on {running_hours:g} running hours in the last {lookback_days} days "
            f"({hours(outliers):g} h of unusual readings removed): usual flow {typical:.1f} m³/h, "
            f"lowest normal flow {lowest_normal:.1f} m³/h. Suggested minimum = {lowest_normal:.1f} − {margin_percent:g}% = {suggested:.1f} m³/h."
        ),
    }
