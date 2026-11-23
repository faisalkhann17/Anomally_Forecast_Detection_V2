"""Data quality firewall and R/Y/B average feature helpers."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from backend.engine.electrical_profile import FORECAST_FIELDS, current_faults
from backend.engine.on_only_aggregation import derive_pump_on


CURRENT_PHASES = ["current_r", "current_y", "current_b"]
VOLTAGE_PHASES = ["voltage_r", "voltage_y", "voltage_b"]
PF_PHASES = ["pf_r", "pf_y", "pf_b"]
POWER_FIELDS = ["total_kw"]
FLOW_FIELDS = ["flowrate"]
SENSOR_FIELDS = CURRENT_PHASES + VOLTAGE_PHASES + PF_PHASES + POWER_FIELDS + FLOW_FIELDS
REQUIRED_FOR_PREDICTION = CURRENT_PHASES + VOLTAGE_PHASES + PF_PHASES + POWER_FIELDS
MAX_FUTURE_SKEW = timedelta(minutes=5)
DEFAULT_MIN_CURRENT_DRAW_A = 0.5
DEFAULT_MIN_POWER_DRAW_KW = 0.1


def _empty_result() -> dict[str, Any]:
    """Create the default passing data-quality result structure."""
    return {
        "status": "PASS",
        "issues": [],
        "invalid_fields": [],
        "missing_fields": [],
        "unavailable_fields": [],
        "hyphen_fields": [],
        "usable_for_baseline": True,
        "usable_for_prediction": True,
    }


def _add_issue(result: dict[str, Any], field: str, message: str, *, invalid: bool = False,
               missing: bool = False, unavailable: bool = True) -> None:
    """Record one field issue and update its classification lists."""
    result["issues"].append(message)
    if invalid and field not in result["invalid_fields"]:
        result["invalid_fields"].append(field)
    if missing and field not in result["missing_fields"]:
        result["missing_fields"].append(field)
    if unavailable and field not in result["unavailable_fields"]:
        result["unavailable_fields"].append(field)


def data_quality_firewall(row: pd.Series | dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Validate one telemetry row without inventing missing sensor values."""
    data = dict(row)
    result = _empty_result()
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))

    timestamp = pd.to_datetime(data.get("_time"), utc=True, errors="coerce")
    if pd.isna(timestamp):
        _add_issue(result, "_time", "Timestamp is missing or malformed.", invalid=True)
    elif timestamp > now_ts + MAX_FUTURE_SKEW:
        _add_issue(result, "_time", "Timestamp is too far in the future.", invalid=True)

    status = str(data.get("pump_status", "")).upper().strip()
    pump_on = status in {"ON", "RUNNING"}
    numeric_voltages = [pd.to_numeric(data.get(field), errors="coerce") for field in VOLTAGE_PHASES]
    finite_voltages = [float(value) for value in numeric_voltages if pd.notna(value)]
    possible_phase_loss = (
        pump_on
        and len(finite_voltages) == len(VOLTAGE_PHASES)
        and sum(value < 10 for value in finite_voltages) == 1
        and max(finite_voltages) >= 50
    )

    # Influx rows are coerced to numbers before validation; the loader records which
    # fields arrived as '-' so they are still judged as hyphens, not as plain blanks.
    marked = data.get("hyphen_fields")
    hyphen_marked = {name.strip() for name in marked.split(",") if name.strip()} if isinstance(marked, str) else set()
    infinite = data.get("nonfinite_fields")
    infinite_marked = {name.strip() for name in infinite.split(",") if name.strip()} if isinstance(infinite, str) else set()
    for field in SENSOR_FIELDS:
        raw = data.get(field)
        if field in hyphen_marked:
            raw = "-"
        if field in infinite_marked:
            _add_issue(result, field, f"{field} contains an infinite value.", invalid=True)
            continue
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            _add_issue(result, field, f"{field} is missing.", missing=True,
                       invalid=field in CURRENT_PHASES or field == "total_kw")
            continue
        if str(raw).strip() == "-":
            if field not in result["hyphen_fields"]:
                result["hyphen_fields"].append(field)
            _add_issue(result, field, f"{field} contains '-' and is unavailable.", invalid=field in REQUIRED_FOR_PREDICTION)
            continue
        value = pd.to_numeric(raw, errors="coerce")
        if pd.isna(value):
            _add_issue(result, field, f"{field} is non-numeric.", invalid=field in REQUIRED_FOR_PREDICTION)
            continue
        value = float(value)
        if not math.isfinite(value):
            _add_issue(result, field, f"{field} contains an infinite value.", invalid=True)
            continue
        if field in CURRENT_PHASES:
            if value < 0:
                _add_issue(result, field, f"{field} cannot be negative.", invalid=True)
        elif field in VOLTAGE_PHASES:
            if value < 0:
                _add_issue(result, field, f"{field} cannot be negative.", invalid=True)
            elif pump_on and value < 10 and not possible_phase_loss:
                _add_issue(result, field, f"{field} looks like a voltage scale mistake.", invalid=True)
        elif field in PF_PHASES:
            if value > 1:
                _add_issue(result, field, "PF appears to be percentage scale; expected 0 to 1.", invalid=True)
            elif value < 0:
                _add_issue(result, field, f"{field} must be between 0 and 1.", invalid=True)
        elif field == "total_kw":
            if value < 0 or value >= 99999:
                _add_issue(result, field, f"{field} is invalid or sentinel.", invalid=True)
        elif field == "flowrate" and value < 0:
            _add_issue(result, field, f"{field} cannot be negative.", invalid=True)

    for group_name, fields in (("current", CURRENT_PHASES), ("voltage", VOLTAGE_PHASES), ("pf", PF_PHASES)):
        available = [
            field for field in fields
            if field not in result["invalid_fields"] and field not in result["missing_fields"] and field not in result["hyphen_fields"]
        ]
        if not available:
            _add_issue(result, group_name, f"All {group_name} phases are unavailable.", invalid=True)

    if result["invalid_fields"]:
        result["status"] = "FAIL"
        result["usable_for_baseline"] = False
        result["usable_for_prediction"] = False
    elif result["unavailable_fields"]:
        result["status"] = "WARN"
    return result


def add_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add phase-average current, voltage and power-factor columns."""
    out = df.copy()
    for fields, target in ((CURRENT_PHASES, "current_avg"),
                           (VOLTAGE_PHASES, "voltage_avg"),
                           (PF_PHASES, "pf_avg")):
        present = [field for field in fields if field in out.columns]
        if present:
            out[target] = out[present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        else:
            out[target] = float("nan")
    return out


def forecast_input_firewall(
    frame: pd.DataFrame,
    profile: dict | None = None,
    *,
    min_current_draw_a: float = DEFAULT_MIN_CURRENT_DRAW_A,
    min_power_draw_kw: float = DEFAULT_MIN_POWER_DRAW_KW,
) -> pd.DataFrame:
    """Mask unusable targets but keep timestamps and state for the forecast model.

    This stops OFF, missing, no-draw and hard-fault rows from becoming normal
    target examples without pretending the pump ran continuously.
    """
    work = frame.copy()
    if work.empty:
        work["_pump_on"] = pd.Series(dtype=bool)
        work["forecast_input_usable"] = pd.Series(dtype=bool)
        work["forecast_input_reason"] = pd.Series(dtype=str)
        work["status_on_no_draw_count"] = pd.Series(dtype="int64")
        work["invalid_on_count"] = pd.Series(dtype="int64")
        return work

    pump_on = derive_pump_on(work)
    usable: list[bool] = []
    reasons: list[str] = []
    no_draw_counts: list[int] = []
    invalid_on_counts: list[int] = []
    for index, row in work.iterrows():
        if not bool(pump_on.loc[index]):
            usable.append(False)
            reasons.append("OFF")
            no_draw_counts.append(0)
            invalid_on_counts.append(0)
            continue
        quality_row = row.copy()
        quality_row["pump_status"] = "ON"
        quality = data_quality_firewall(quality_row)
        complete = all(pd.notna(pd.to_numeric(row.get(field), errors="coerce")) for field in FORECAST_FIELDS)
        if not quality["usable_for_prediction"] or not complete:
            usable.append(False)
            reasons.append("INVALID")
            no_draw_counts.append(0)
            invalid_on_counts.append(1)
            continue
        currents = pd.to_numeric(row.reindex(CURRENT_PHASES), errors="coerce")
        total_kw = pd.to_numeric(row.get("total_kw"), errors="coerce")
        if (currents <= min_current_draw_a).all() and total_kw <= min_power_draw_kw:
            usable.append(False)
            reasons.append("STATUS_ON_NO_DRAW")
            no_draw_counts.append(1)
            invalid_on_counts.append(0)
            continue
        if profile is not None and current_faults(row, profile):
            usable.append(False)
            reasons.append("HARD_FAULT")
            no_draw_counts.append(0)
            invalid_on_counts.append(1)
            continue
        usable.append(True)
        reasons.append("USABLE")
        no_draw_counts.append(0)
        invalid_on_counts.append(0)
    work["_pump_on"] = pump_on.to_numpy(dtype=bool)
    work["forecast_input_usable"] = usable
    work["forecast_input_reason"] = reasons
    work["status_on_no_draw_count"] = no_draw_counts
    work["invalid_on_count"] = invalid_on_counts
    work.loc[~work["forecast_input_usable"], list(FORECAST_FIELDS)] = float("nan")
    return work


def phase_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Build dashboard-safe R/Y/B phase values and averages."""
    data = dict(row)

    def build(prefix: str) -> dict[str, Any]:
        """Build one measurement group's phase and average payload."""
        payload = {}
        phases = []
        values = []
        for phase in ("r", "y", "b"):
            key = f"{prefix}_{phase}"
            value = pd.to_numeric(data.get(key), errors="coerce")
            payload[phase] = None if pd.isna(value) else round(float(value), 3)
            if not pd.isna(value):
                phases.append(phase)
                values.append(float(value))
        avg_key = f"{prefix}_avg"
        avg = pd.to_numeric(data.get(avg_key), errors="coerce")
        if pd.isna(avg) and values:
            avg = sum(values) / len(values)
        payload["avg"] = None if pd.isna(avg) else round(float(avg), 3)
        payload["available_phases"] = phases
        return payload

    return {
        "current": build("current"),
        "voltage": build("voltage"),
        "pf": build("pf"),
        "power": {"total_kw": _round_or_none(data.get("total_kw"))},
        "flow": {"flowrate": _round_or_none(data.get("flowrate"))},
    }


def _round_or_none(value: Any) -> float | None:
    """Return a rounded number or None for unavailable input."""
    value = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(value) else round(float(value), 3)
