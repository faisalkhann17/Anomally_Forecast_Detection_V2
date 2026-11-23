"""Direct short-cycling detection from recent energy-meter operating events."""
from __future__ import annotations

from statistics import median

import pandas as pd


def _operating_state(value: object, total_kw: object, current_b: object,
                     previous: str | None, *, on_power_kw: float,
                     off_power_kw: float, on_current_a: float,
                     off_current_a: float) -> str | None:
    """Resolve a debounced ON/OFF state from power, tag and current evidence."""
    power = pd.to_numeric(total_kw, errors="coerce")
    if pd.notna(power):
        if float(power) >= on_power_kw:
            return "ON"
        if float(power) <= off_power_kw:
            return "OFF"

    tag = str(value or "").upper().strip()
    if tag in {"ON", "RUNNING"}:
        return "ON"
    if tag in {"OFF", "OFFLINE", "STOPPED"}:
        return "OFF"

    current = pd.to_numeric(current_b, errors="coerce")
    if pd.notna(current):
        if float(current) >= on_current_a:
            return "ON"
        if float(current) <= off_current_a:
            return "OFF"
    return previous


def evaluate_short_cycling(
    frame: pd.DataFrame,
    *,
    lookback_minutes: int = 60,
    max_starts: int = 4,
    min_on_seconds: int = 30,
    min_off_seconds: int = 30,
    max_gap_seconds: int = 300,
    min_observation_minutes: int = 45,
    on_power_kw: float = 0.2,
    off_power_kw: float = 0.1,
    on_current_a: float = 1.0,
    off_current_a: float = 0.5,
) -> dict:
    """Count debounced OFF-to-ON transitions in a rolling operating window."""
    base = {
        "evaluated": False,
        "active": False,
        "starts_in_window": 0,
        "allowed_starts": int(max_starts),
        "lookback_minutes": int(lookback_minutes),
        "median_on_minutes": None,
        "observation_minutes": 0.0,
        "reason": None,
    }
    if frame is None or frame.empty or "_time" not in frame:
        return {**base, "reason": "Recent ON/OFF event history is unavailable."}

    work = frame.copy()
    work["_time"] = pd.to_datetime(work["_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["_time"]).sort_values("_time").drop_duplicates(
        "_time", keep="last"
    )
    if work.empty:
        return {**base, "reason": "Recent ON/OFF event history is unavailable."}

    states: list[str | None] = []
    previous = None
    for _, row in work.iterrows():
        state = _operating_state(
            row.get("pump_status"), row.get("total_kw"), row.get("current_b"), previous,
            on_power_kw=on_power_kw, off_power_kw=off_power_kw,
            on_current_a=on_current_a, off_current_a=off_current_a,
        )
        states.append(state)
        if state is not None:
            previous = state
    work["_state"] = states
    work = work.dropna(subset=["_state"])
    if len(work) < 2:
        return {**base, "reason": "Not enough valid operating-state samples."}

    observation_minutes = (
        work["_time"].iloc[-1] - work["_time"].iloc[0]
    ).total_seconds() / 60.0
    base["observation_minutes"] = round(max(0.0, observation_minutes), 2)
    if observation_minutes < min_observation_minutes:
        return {
            **base,
            "reason": (
                f"Only {observation_minutes:.1f} minutes of operating-state history "
                f"is available; {min_observation_minutes} minutes is required."
            ),
        }

    if "total_kw" not in work.columns:
        return {
            **base,
            "reason": "No zero-kW operating episode is present in the available history.",
        }
    power = pd.to_numeric(work["total_kw"], errors="coerce")
    zero_power = power.between(0.0, off_power_kw, inclusive="both")
    if not bool(zero_power.any()):
        return {
            **base,
            "reason": "No zero-kW operating episode is present in the available history.",
        }

    # A cycle must come from real energy-meter readings. Missing power values
    # and status-tag chatter are not allowed to manufacture transitions.
    work = work.loc[power.notna()].copy()
    if len(work) < 2:
        return {**base, "reason": "Not enough valid power samples."}

    runs: list[dict] = []
    for _, row in work.iterrows():
        timestamp = row["_time"]
        state = row["_state"]
        if not runs:
            runs.append({"state": state, "start": timestamp, "end": timestamp,
                         "connected": False})
            continue
        gap_seconds = (timestamp - runs[-1]["end"]).total_seconds()
        if state != runs[-1]["state"] or gap_seconds > max_gap_seconds:
            runs.append({
                "state": state,
                "start": timestamp,
                "end": timestamp,
                "connected": gap_seconds <= max_gap_seconds,
            })
        else:
            runs[-1]["end"] = timestamp

    has_zero_power_episode = any(
        run["state"] == "OFF"
        and (run["end"] - run["start"]).total_seconds() >= min_off_seconds
        for run in runs
    )
    if not has_zero_power_episode:
        return {
            **base,
            "observation_minutes": round(max(0.0, observation_minutes), 2),
            "reason": (
                "Zero-kW samples are present, but no validated zero-kW episode "
                f"lasted at least {min_off_seconds} seconds."
            ),
        }

    reference = work["_time"].iloc[-1]
    cutoff = reference - pd.to_timedelta(float(lookback_minutes), unit="min")
    starts = []
    on_durations = []
    for index in range(1, len(runs)):
        current = runs[index]
        previous_run = runs[index - 1]
        if current["state"] != "ON" or previous_run["state"] != "OFF":
            continue
        if not current["connected"]:
            continue
        off_seconds = (previous_run["end"] - previous_run["start"]).total_seconds()
        on_seconds = (current["end"] - current["start"]).total_seconds()
        if off_seconds < min_off_seconds or on_seconds < min_on_seconds:
            continue
        if cutoff <= current["start"] <= reference:
            starts.append(current["start"])
            on_durations.append(on_seconds / 60.0)

    starts_count = len(starts)
    median_on = round(float(median(on_durations)), 2) if on_durations else None
    result = {
        **base,
        "evaluated": True,
        "starts_in_window": starts_count,
        "median_on_minutes": median_on,
        "reason": None,
    }
    if starts_count <= max_starts:
        return result

    run_text = (
        f" Each run lasted about {median_on:.1f} minutes."
        if median_on is not None else ""
    )
    result["active"] = True
    result["fault"] = {
        "code": "SHORT_CYCLING",
        "label": "Short Cycling",
        "detection_source": "OPERATING_EVENT_HISTORY",
        "evidence": (
            f"The pump started {starts_count} times in the last "
            f"{lookback_minutes} minutes; the limit is {max_starts}.{run_text}"
        ),
        "starts_in_window": starts_count,
        "allowed_starts": int(max_starts),
        "lookback_minutes": int(lookback_minutes),
        "median_on_minutes": median_on,
    }
    return result
