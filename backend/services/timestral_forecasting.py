"""Run local Timestral inference on cleaned pump history, conditional on pump ON."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import settings
from backend.engine.data_quality import forecast_input_firewall
from backend.engine.electrical_profile import FORECAST_FIELDS
from backend.engine.on_only_aggregation import aggregate_on_only


class TimestralForecastService:
    """Convert regular pump history into quantile forecasts on the configured grid."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        context_points: int = 2016,
        horizon_points: int = 72,
        interval: str = "5min",
        interval_minutes: int = 5,
        minimum_history_points: int = 288,
    ) -> None:
        """Configure one local CPU Timestral service for a grid and horizon."""
        self.checkpoint = checkpoint
        self.context_points = context_points
        self.horizon_points = horizon_points
        self.interval = interval
        self.interval_minutes = interval_minutes
        self.minimum_history_points = minimum_history_points
        self.forecaster = None
        self.load_error: str | None = None

    def _load(self) -> None:
        """Lazily load the local checkpoint and retain any load error."""
        if self.forecaster is not None or self.load_error is not None:
            return
        try:
            import torch
            from timestral import TimestralForecaster

            if settings.FORECAST_TORCH_THREADS > 0:
                torch.set_num_threads(settings.FORECAST_TORCH_THREADS)
            self.forecaster = TimestralForecaster.from_pretrained(
                self.checkpoint, device="cpu", local_files_only=True
            )
        except Exception as exc:
            self.load_error = str(exc)

    def _recommended_horizon(self, calendar_points: int) -> int:
        """Grow the horizon as the pump's calendar history accumulates."""
        if self.interval_minutes == 5:
            if calendar_points < 864:
                return min(12, self.horizon_points)
            if calendar_points < 2016:
                return min(36, self.horizon_points)
            return min(72, self.horizon_points)
        if calendar_points < 336:
            return min(24, self.horizon_points)
        if calendar_points < 720:
            return min(48, self.horizon_points)
        if calendar_points < 1440:
            return min(72, self.horizon_points)
        if calendar_points < 2160:
            return min(120, self.horizon_points)
        return min(360, self.horizon_points)

    def _regular_history(self, frame: pd.DataFrame, profile: dict | None = None) -> tuple[pd.DataFrame, bool]:
        """Validate, mask and aggregate telemetry onto the configured grid."""
        if self.interval_minutes == 60 and {"_time", "_pump_on", "on_sample_count", *FORECAST_FIELDS}.issubset(frame.columns):
            values = frame.copy()
            values["_time"] = pd.to_datetime(values["_time"], utc=True, errors="coerce")
            values = values.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last").set_index("_time")
            eligible = pd.to_numeric(values["on_sample_count"], errors="coerce").ge(settings.FORECAST_ON_MIN_SAMPLES_HOURLY)
            values.loc[~eligible, list(FORECAST_FIELDS)] = np.nan
            values["_pump_on"] = values["_pump_on"].fillna(0).astype(bool) & eligible
            return values, bool(values["_pump_on"].iloc[-1]) if not values.empty else False
        min_on_samples = (
            settings.FORECAST_ON_MIN_SAMPLES_5MIN
            if self.interval_minutes < 60
            else settings.FORECAST_ON_MIN_SAMPLES_HOURLY
        )
        values = aggregate_on_only(
            forecast_input_firewall(frame, profile),
            interval=self.interval,
            target_fields=FORECAST_FIELDS,
            min_on_samples=min_on_samples,
        )
        latest_on = bool(values["_pump_on"].iloc[-1]) if not values.empty else False
        return values, latest_on

    @staticmethod
    def _ordered_prediction(values: np.ndarray, levels: list[float]) -> tuple[dict, list[float], bool]:
        """Sort quantiles, prevent crossings and build the field payload."""
        order = np.argsort(np.asarray(levels, dtype=float))
        sorted_levels = [float(levels[index]) for index in order]
        ordered = values[:, :, order]
        corrected = not np.all(np.diff(ordered, axis=2) >= 0)
        ordered = np.maximum.accumulate(ordered, axis=2)
        prediction: dict[str, dict] = {}
        for index, field in enumerate(FORECAST_FIELDS):
            quantiles = {
                f"{level:.3f}": [round(float(value), 4) for value in ordered[index, :, level_index]]
                for level_index, level in enumerate(sorted_levels)
            }
            aliases = {
                name: quantiles[f"{target:.3f}"]
                for name, target in (("q05", .05), ("q50", .50), ("q95", .95))
                if f"{target:.3f}" in quantiles
            }
            prediction[field] = {"quantiles": quantiles, **aliases}
        return prediction, sorted_levels, corrected

    def forecast(self, frame: pd.DataFrame, profile: dict | None = None) -> dict:
        """Check usable history, forecast future ON readings and return all quantiles."""
        prepared, currently_on = self._regular_history(frame, profile)
        if prepared.empty:
            return {"state": "NO_TELEMETRY"}
        history = len(prepared)
        usable_history = int(prepared[list(FORECAST_FIELDS)].notna().all(axis=1).sum())
        if not currently_on:
            return {"state": "OFF_OR_NO_ACTIVE_SESSION", "history_points": history}
        if usable_history < self.minimum_history_points:
            return {
                "state": "MODEL_WARMUP",
                "history_points": usable_history,
                "calendar_points": history,
                "required_points": self.minimum_history_points,
            }
        self._load()
        if self.forecaster is None:
            return {"state": "MODEL_UNAVAILABLE", "detail": self.load_error}
        # Calendar history controls how far ahead the advisory may extend.
        # Usable ON observations remain the independent warm-up gate above.
        horizon_points = self._recommended_horizon(history)
        context = prepared.tail(self.context_points)
        targets = context[list(FORECAST_FIELDS)].apply(pd.to_numeric, errors="coerce")
        on_context = context["_pump_on"].fillna(0).round().to_numpy(dtype=np.float32)
        observed_on_points = int((on_context >= 1).sum())
        if observed_on_points < 32:
            return {
                "state": "MODEL_WARMUP",
                "history_points": history,
                "on_history_points": observed_on_points,
                "required_on_points": 32,
            }
        known_on = np.concatenate([on_context, np.ones(horizon_points, dtype=np.float32)])
        forecast = self.forecaster.forecast(
            targets.to_numpy(dtype=np.float32, copy=True).T[None, :, :],
            horizon_points,
            categorical_known_covariates=known_on[None, None, :],
        )
        values = forecast.quantiles.detach().cpu().numpy()[0]
        levels = [float(level) for level in forecast.quantile_levels]
        prediction, ordered_levels, quantile_ordering_enforced = self._ordered_prediction(values, levels)
        return {
            "state": "FORECAST_READY",
            "history_points": usable_history,
            "calendar_points": history,
            "on_history_points": observed_on_points,
            "history_duration_hours": round(usable_history * self.interval_minutes / 60, 1),
            "interval_minutes": self.interval_minutes,
            "horizon_minutes": horizon_points * self.interval_minutes,
            "conditional_on_pump_on": True,
            "quantile_levels": ordered_levels,
            "quantile_ordering_enforced": quantile_ordering_enforced,
            "prediction": prediction,
        }
