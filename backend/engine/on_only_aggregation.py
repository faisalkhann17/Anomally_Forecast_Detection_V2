"""Resample status-ON, usable telemetry without diluting means with OFF rows."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CURRENT_FIELDS = ("current_r", "current_y", "current_b")
DEFAULT_MIN_CURRENT_DRAW_A = 0.5
DEFAULT_MIN_POWER_DRAW_KW = 0.1


def derive_pump_on(frame: pd.DataFrame) -> pd.Series:
    """Return the authoritative running state supplied by the pump status tag."""
    status = frame.get("pump_status", pd.Series("unknown", index=frame.index))
    return status.astype(str).str.upper().str.strip().isin(["ON", "RUNNING"])


def aggregate_on_only(
    frame: pd.DataFrame,
    *,
    interval: str,
    target_fields: Iterable[str],
    min_on_samples: int = 1,
) -> pd.DataFrame:
    """Build 5-minute/hourly buckets from usable ON rows and preserve empty buckets as NaN."""
    if min_on_samples < 1:
        raise ValueError("min_on_samples must be at least one")
    if "_time" not in frame:
        raise ValueError("frame must contain _time")

    work = frame.copy()
    work["_time"] = pd.to_datetime(work["_time"], utc=True, errors="coerce")
    work = work.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last")
    fields = list(target_fields)
    if work.empty:
        return pd.DataFrame(columns=[
            *fields, "_pump_on", "on_fraction", "on_sample_count", "status_on_samples",
            "usable_on_samples", "invalid_on_samples", "no_draw_samples",
        ])

    # A status-ON row is counted only after the forecast firewall validates its draw.
    status_on = derive_pump_on(work)
    if "forecast_input_usable" in work:
        usable_on = status_on & work["forecast_input_usable"].fillna(False).astype(bool)
        no_draw = work.get("forecast_input_reason", pd.Series("", index=work.index)).eq("STATUS_ON_NO_DRAW")
    else:
        numeric_targets = pd.DataFrame({
            field: pd.to_numeric(work.get(field), errors="coerce") for field in fields
        })
        complete = numeric_targets.notna().all(axis=1)
        current_fields = [field for field in CURRENT_FIELDS if field in work]
        currents = pd.DataFrame({
            field: pd.to_numeric(work[field], errors="coerce") for field in current_fields
        })
        total_kw = pd.to_numeric(work.get("total_kw"), errors="coerce")
        no_draw = (
            currents.notna().all(axis=1)
            & (currents <= DEFAULT_MIN_CURRENT_DRAW_A).all(axis=1)
            & total_kw.notna()
            & (total_kw <= DEFAULT_MIN_POWER_DRAW_KW)
        ) if current_fields else pd.Series(False, index=work.index)
        usable_on = status_on & complete & ~no_draw
    invalid_on = status_on & ~usable_on & ~no_draw
    indexed = work.set_index("_time")
    status_indexed = pd.Series(status_on.to_numpy(dtype=bool), index=indexed.index, name="status_on")
    usable_indexed = pd.Series(usable_on.to_numpy(dtype=bool), index=indexed.index, name="_pump_on")
    invalid_indexed = pd.Series(invalid_on.to_numpy(dtype=bool), index=indexed.index, name="invalid_on")
    no_draw_indexed = pd.Series(no_draw.to_numpy(dtype=bool), index=indexed.index, name="no_draw")
    total_samples = indexed.resample(interval).size()
    usable_on_samples = usable_indexed.astype("int64").resample(interval).sum()
    eligible = usable_on_samples >= min_on_samples

    result = pd.DataFrame(index=total_samples.index)
    for field in fields:
        source = pd.to_numeric(indexed.get(field, pd.Series(np.nan, index=indexed.index)), errors="coerce")
        result[field] = source.where(usable_indexed).resample(interval).mean().where(eligible)

    result["_pump_on"] = usable_indexed.astype(float).resample(interval).max().fillna(0.0)
    result["status_on_samples"] = status_indexed.astype("int64").resample(interval).sum().astype("int64")
    result["usable_on_samples"] = usable_on_samples.astype("int64")
    result["invalid_on_samples"] = invalid_indexed.astype("int64").resample(interval).sum().astype("int64")
    result["no_draw_samples"] = no_draw_indexed.astype("int64").resample(interval).sum().astype("int64")
    # Coverage describes usable ON readings, not merely rows tagged ON.
    result["on_sample_count"] = result["usable_on_samples"]
    result["on_fraction"] = (usable_on_samples / total_samples.replace(0, np.nan)).astype(float)
    return result
