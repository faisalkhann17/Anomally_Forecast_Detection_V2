"""Coordinate live rule checks, cached Timestral forecasts and maintenance jobs."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.engine.short_cycling import evaluate_short_cycling
from backend.engine.health_score import HealthSmoother, calculate_smoothed_health, profile_health_baseline
from backend.engine.data_quality import (
    data_quality_firewall,
    phase_payload,
)
from backend.services.influx_service import InfluxService
from backend.services.pump_profile_repository import ProfileStoreUnavailable, create_profile_repository
from backend.services.maintenance_baseline_repository import MaintenanceBaselineRepository
from backend.services.issue_event_repository import MongoIssueEventRepository
from backend.services.timestral_forecasting import TimestralForecastService
from backend.engine.electrical_profile import current_faults, enriched_frame, forecast_risk
from backend.engine.electrical_maintenance import analyze_electrical_maintenance
from backend.engine.pump_performance_features import analyze_performance
from backend.engine.slow_degradation_patterns import analyze_combined_slow_degradation, corroborate_common_supply
from backend.engine.live_anomaly_context import (
    LIVE_CONTEXT_ROWS,
    current_session_summary,
    immediate_faults,
    long_term_normal,
    long_term_shift_warnings,
    robust_shift_warnings,
)
from backend.config import settings

logger = logging.getLogger(__name__)

# Existing API imports these caches directly.
_in_memory_cache: dict = {}
_pump_states: dict = {}
_operating_event_states: dict = {}
def _is_physically_off(row: pd.Series) -> bool:
    """Treat explicit OFF status or near-zero electrical draw as stopped."""
    if str(row.get("pump_status", "")).upper().strip() in ("OFF", "OFFLINE"):
        return True
    fields = pd.to_numeric(pd.Series({
        key: row.get(key) for key in ("current_b", "current_r", "current_y", "total_kw")
    }), errors="coerce")
    return bool(fields.notna().all() and (fields.iloc[:3] <= .5).all()
                and fields.iloc[3] <= .1)



class PredictionWorker:
    """Keep request-time anomaly checks separate from scheduled historical analysis."""

    def __init__(self):
        """Create repositories, forecast services, caches and job controls."""
        self.influx_service = InfluxService()
        self.is_running = False
        self.load_error = None
        self.pump_discovery_error: str | None = None
        # Engineer profiles (approved limits) and their history live in MongoDB only.
        audit_anchor = Path(__file__).resolve().parent.parent / "artifacts" / "pump_profiles.json"
        self.profile_repository = create_profile_repository(audit_anchor)
        checkpoint = Path(__file__).resolve().parent.parent / "artifacts" / "timestral" / "checkpoints" / "Timestral-123M"
        # The short 5-minute forecast path is intentionally disabled. PumpWatch
        # uses one hourly Timestral forecast for advisory long-horizon trends.
        self.hourly_timestral = TimestralForecastService(
            checkpoint, context_points=2160, horizon_points=240,
            interval="1h", interval_minutes=60, minimum_history_points=48,
        )
        # Scheduled results are reused by dashboard reads instead of recomputing per request.
        self.forecast_cache: dict[str, dict] = {}
        self.forecast_cache_minutes = settings.FORECAST_REFRESH_MINUTES
        self.maintenance_cache: dict[str, dict] = {}
        # Per-pump usual level (7-day median of ON hours), refreshed by the maintenance job.
        self.long_term_normals: dict[str, dict] = {}
        self.maintenance_cache_minutes = settings.MAINTENANCE_REFRESH_MINUTES
        self.forecast_task: asyncio.Task | None = None
        self.maintenance_task: asyncio.Task | None = None
        self._historical_job_lock = asyncio.Lock()
        self._first_forecast_cycle_done = asyncio.Event()
        maintenance_path = Path(__file__).resolve().parent.parent / "artifacts" / "electrical_maintenance_baselines.json"
        self.maintenance_baselines = MaintenanceBaselineRepository(maintenance_path)
        self.issue_event_repository = None
        self.issue_event_store_error = None
        self.issue_event_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pumpwatch-events")
        try:
            self.issue_event_repository = MongoIssueEventRepository(
                settings.PROFILE_MONGODB_URI,
                settings.PROFILE_MONGODB_DATABASE,
                settings.MONGODB_EVENT_COLLECTION,
                settings.MONGODB_TIMEOUT_MS,
            )
        except Exception as exc:
            self.issue_event_store_error = str(exc)
            logger.warning("Issue event store unavailable: %s", exc)
        self.health_smoother = HealthSmoother(window=5)

    async def initialize_pump(self, pump_id: str, force_enroll: bool = False):
        """Build the first live status payload for a newly discovered pump."""
        await self._refresh_profile_pipeline(pump_id)

    @staticmethod
    def _is_known_pump(pump_id: str) -> bool:
        """True when the last Influx pump list included this ID."""
        from backend.engine import config
        return pump_id in config.WORKING_PUMPS

    async def refresh_pump(self, pump_id: str):
        """Refresh latest-reading faults without running historical Timestral inference."""
        await self._refresh_profile_pipeline(pump_id)

    def invalidate_forecast(self, pump_id: str) -> None:
        """Discard results based on old limits when an engineering profile changes."""
        self.forecast_cache.pop(pump_id, None)
        self.maintenance_cache.pop(pump_id, None)
        _in_memory_cache.pop(pump_id, None)
        # Scores computed against the old limits must not be smoothed into new ones.
        self.health_smoother.reset(pump_id)

    def _cached_forecast(self, pump_id: str) -> dict | None:
        """Return a forecast while its configured cache lifetime is valid."""
        cached = self.forecast_cache.get(pump_id)
        if not cached:
            return None
        age = datetime.now(timezone.utc) - cached["created_at"]
        if age >= pd.Timedelta(minutes=self.forecast_cache_minutes):
            return None
        return cached["payload"]

    def _cached_maintenance(self, pump_id: str) -> dict | None:
        """Return maintenance guidance while its cache entry is fresh."""
        cached = self.maintenance_cache.get(pump_id)
        if not cached:
            return None
        age = datetime.now(timezone.utc) - cached["created_at"]
        if age >= pd.Timedelta(minutes=self.maintenance_cache_minutes):
            return None
        return cached["payload"]

    async def _refresh_profile_pipeline(self, pump_id: str) -> None:
        """Check recent telemetry and approved limits; attach cached future guidance."""
        try:
            profile = self.profile_repository.get(pump_id)
        except ProfileStoreUnavailable as exc:
            self._store_profile_result(pump_id, pd.DataFrame(), None, "DATA_ERROR", detail=str(exc))
            return
        try:
            recent_frame = await asyncio.to_thread(
                self.influx_service.get_recent_wide_dataframe, pump_id, "-1h", 10
            )
        except Exception as exc:
            logger.exception("Telemetry fetch failed for %s", pump_id)
            self._store_profile_result(pump_id, pd.DataFrame(), profile, "DATA_ERROR", detail=str(exc))
            return
        if recent_frame.empty:
            if profile is None and not self._is_known_pump(pump_id):
                # No profile, never listed by Influx and no telemetry: an unknown ID.
                # Leave it out of the cache so the API answers 404 instead of 200.
                _in_memory_cache.pop(pump_id, None)
                return
            self._store_profile_result(pump_id, recent_frame, profile, "DATA_ERROR", detail="No recent telemetry is available.")
            return
        recent_frame = enriched_frame(recent_frame)
        recent_frame["_time"] = pd.to_datetime(recent_frame["_time"], utc=True, errors="coerce")
        recent_frame = recent_frame.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last")
        latest = recent_frame.iloc[-1]
        _pump_states[pump_id] = recent_frame
        _operating_event_states[pump_id] = recent_frame
        if profile is None:
            self._store_profile_result(pump_id, recent_frame, None, "CONFIGURATION_REQUIRED")
            return
        if profile.get("approval_status") != "APPROVED":
            self._store_profile_result(pump_id, recent_frame, profile, "PENDING_APPROVAL")
            return
        quality = data_quality_firewall(latest)
        if quality["status"] == "FAIL":
            self._store_profile_result(pump_id, recent_frame, profile, "DATA_QUALITY_ERROR", data_quality=quality)
            return
        data_age_seconds = max(0.0, (datetime.now(timezone.utc) - latest["_time"].to_pydatetime()).total_seconds())
        if data_age_seconds > settings.MAX_TELEMETRY_AGE_SECONDS:
            self._store_profile_result(pump_id, recent_frame, profile, "STALE_DATA", data_quality=quality, data_age_seconds=data_age_seconds)
            return
        pump_status = str(latest.get("pump_status") or "").upper()
        if pump_status in {"OFF", "OFFLINE"} or _is_physically_off(latest):
            self._store_profile_result(pump_id, recent_frame, profile, "OFF", data_quality=quality)
            return
        try:
            frame = await asyncio.to_thread(
                self.influx_service.get_latest_live_on_dataframe,
                pump_id,
                LIVE_CONTEXT_ROWS,
                bounded=True,
            )
            if frame.empty:
                frame = recent_frame
        except Exception:
            logger.exception("Live 350-row context unavailable for %s; using recent telemetry", pump_id)
            frame = recent_frame
        frame = enriched_frame(frame)
        frame["_time"] = pd.to_datetime(frame["_time"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last")
        _pump_states[pump_id] = frame
        session_reading, session_rows = current_session_summary(frame)
        analysis_reading = session_reading if session_reading is not None else latest
        faults = immediate_faults(latest, profile)
        persistent_faults = current_faults(analysis_reading, profile) if session_reading is not None else []
        existing_codes = {fault["code"] for fault in faults}
        faults.extend(fault for fault in persistent_faults if fault["code"] not in existing_codes)
        early_warnings = []
        if not faults:
            early_warnings = robust_shift_warnings(frame, session_reading, profile)
            seen = {warning["code"] for warning in early_warnings}
            early_warnings.extend(
                warning for warning in long_term_shift_warnings(
                    session_reading, self._long_term_normal(pump_id), profile,
                    percent=settings.EARLY_WARNING_LONG_TERM_PERCENT,
                ) if warning["code"] not in seen
            )
        cycling = {}
        if not faults:
            try:
                cycling_frame = await asyncio.to_thread(
                    self.influx_service.get_recent_wide_dataframe,
                    pump_id,
                    f"-{settings.SHORT_CYCLING_LOOKBACK_MINUTES}m",
                    None,
                )
                if not cycling_frame.empty:
                    frame = enriched_frame(cycling_frame)
                    frame["_time"] = pd.to_datetime(frame["_time"], utc=True, errors="coerce")
                    frame = frame.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last")
                    _pump_states[pump_id] = frame
                    _operating_event_states[pump_id] = frame
                    cycling = evaluate_short_cycling(
                        frame,
                        lookback_minutes=settings.SHORT_CYCLING_LOOKBACK_MINUTES,
                        max_starts=settings.SHORT_CYCLING_MAX_STARTS,
                        min_on_seconds=settings.SHORT_CYCLING_MIN_ON_SECONDS,
                        min_off_seconds=settings.SHORT_CYCLING_MIN_OFF_SECONDS,
                        max_gap_seconds=settings.SHORT_CYCLING_MAX_GAP_SECONDS,
                        min_observation_minutes=settings.SHORT_CYCLING_MIN_OBSERVATION_MINUTES,
                    )
                    if cycling.get("active"):
                        faults.append(cycling["fault"])
            except Exception as exc:
                logger.warning("Short-cycling query unavailable for %s: %s", pump_id, exc)
                cycling = {"active": False, "state": "UNAVAILABLE", "detail": str(exc)}

        forecast = self._cached_forecast(pump_id) or {"state": "FORECAST_PENDING"}
        maintenance = self._cached_maintenance(pump_id) or {"state": "MAINTENANCE_PENDING"}
        if faults:
            forecast = {"state": "SKIPPED_CURRENT_FAULT"}
            maintenance = {
                "state": "SUPPRESSED_CURRENT_FAULT",
                "detail": "Slow-degradation assessment is paused while a live engineering fault is active.",
                "signals": [],
            }
        risk_source = forecast
        risk = forecast_risk(risk_source["prediction"], profile, risk_source["interval_minutes"]) if risk_source.get("state") == "FORECAST_READY" else None
        if risk is not None:
            risk["forecast_horizon_hours"] = risk_source["horizon_minutes"] // 60
        detail = None
        if session_reading is None:
            detail = f"Collecting stable ON context ({session_rows}/5 valid recent readings); immediate phase-loss rules remain active."
        self._store_profile_result(
            pump_id,
            frame,
            profile,
            "ANOMALY" if faults else "NORMAL",
            detail=detail,
            data_quality=quality,
            faults=faults,
            cycling=cycling,
            forecast=forecast,
            risk=risk,
            maintenance=maintenance,
            analysis_reading=analysis_reading,
            early_warnings=early_warnings,
            live_session_rows=session_rows,
        )

    async def _forecast_for(self, pump_id: str, profile: dict) -> dict:
        """Fetch hourly history, run long-horizon Timestral and cache it."""
        cached = self._cached_forecast(pump_id)
        if cached is not None:
            return cached
        try:
            try:
                hourly = await asyncio.to_thread(self.influx_service.refresh_hourly_forecast_cache, pump_id, profile)
            except Exception:
                logger.exception("Hourly cache unavailable for %s; using the raw-history fallback", pump_id)
                hourly = await asyncio.to_thread(self.influx_service.get_hourly_forecast_dataframe, pump_id, "-90d", profile)
            payload = await asyncio.to_thread(self.hourly_timestral.forecast, hourly, profile) if not hourly.empty else {
                "state": "MODEL_WARMUP",
                "history_points": 0,
                "required_points": self.hourly_timestral.minimum_history_points,
            }
        except Exception as exc:
            logger.warning("Hourly forecast unavailable for %s: %s", pump_id, exc)
            payload = {"state": "MODEL_UNAVAILABLE", "detail": str(exc)}
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        self.forecast_cache[pump_id] = {"created_at": datetime.now(timezone.utc), "payload": payload}
        return payload

    async def _approved_pump_ids(self) -> list[str]:
        """Return pumps whose engineering profiles are approved (no Influx query)."""
        try:
            profiles = await asyncio.to_thread(self.profile_repository.load_all)
        except Exception:
            logger.exception("Profile store unavailable; skipping this scheduled cycle")
            return []
        return sorted(
            pump_id for pump_id, profile in profiles.items()
            if (profile or {}).get("approval_status") == "APPROVED"
        )

    def _long_term_normal(self, pump_id: str) -> dict | None:
        """Return the pump's usual level while it is fresh enough to compare against."""
        entry = self.long_term_normals.get(pump_id)
        if not entry:
            return None
        age = datetime.now(timezone.utc) - entry["computed_at"]
        if age > pd.Timedelta(hours=settings.EARLY_WARNING_LONG_TERM_MAX_AGE_HOURS):
            return None
        return entry

    async def refresh_electrical_maintenance(self, pump_id: str) -> dict:
        """Assess measured electrical, water-delivery and combined trends off-request."""
        profile = self.profile_repository.get(pump_id)
        if not profile or profile.get("approval_status") != "APPROVED":
            return {"state": "CONFIGURATION_REQUIRED", "signals": []}
        profile_version = int(profile.get("version", 0))
        baseline = self.maintenance_baselines.get(pump_id, profile_version)
        try:
            raw_history = await asyncio.to_thread(
                self.influx_service.get_wide_dataframe,
                pump_id,
                f"-{settings.MAINTENANCE_HISTORY_DAYS}d",
            )
            hourly = await asyncio.to_thread(
                self.influx_service.prepare_hourly_forecast_dataframe,
                raw_history,
                profile,
            )
            normal = long_term_normal(hourly, days=settings.EARLY_WARNING_LONG_TERM_DAYS)
            if normal:
                self.long_term_normals[pump_id] = {**normal, "computed_at": datetime.now(timezone.utc)}
            forecast = self._cached_forecast(pump_id) or {}
            risk = forecast.get("future_electrical_risk")
            if risk is None:
                risk = (_in_memory_cache.get(pump_id) or {}).get("future_electrical_risk")
            result, new_baseline = await asyncio.to_thread(
                analyze_electrical_maintenance,
                hourly,
                profile,
                baseline,
                baseline_points=settings.MAINTENANCE_BASELINE_HOURLY_POINTS,
                recent_points=settings.MAINTENANCE_RECENT_HOURLY_POINTS,
                persistence_points=settings.MAINTENANCE_PERSISTENCE_HOURLY_POINTS,
                forecast_risk=risk,
            )
            performance = await asyncio.to_thread(analyze_performance, raw_history)
            result["performance_degradation"] = performance
            result["combined_slow_degradation"] = await asyncio.to_thread(
                analyze_combined_slow_degradation, result, performance, raw_history, profile
            )
            if result.get("baseline_created") and new_baseline:
                self.maintenance_baselines.save(pump_id, profile_version, new_baseline)
        except Exception as exc:
            logger.exception("Electrical maintenance assessment failed for %s", pump_id)
            result = {"state": "MAINTENANCE_UNAVAILABLE", "detail": str(exc), "signals": []}

        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        self.maintenance_cache[pump_id] = {"created_at": datetime.now(timezone.utc), "payload": result}
        corroborate_common_supply({
            cached_pump_id: cached["payload"]
            for cached_pump_id, cached in self.maintenance_cache.items()
        })
        current = _in_memory_cache.get(pump_id)
        if current and current.get("state") == "NORMAL":
            current["electrical_maintenance"] = result
        self._record_slow_trends(pump_id, result)
        # Corroboration may have relabelled other pumps' supply issues; keep their history in step.
        for other_id, cached in self.maintenance_cache.items():
            combined = (cached.get("payload") or {}).get("combined_slow_degradation") or {}
            if other_id != pump_id and any(
                signal.get("code") == "POSSIBLE_SUPPLY_SIDE_DEGRADATION" for signal in combined.get("signals", [])
            ):
                self._record_slow_trends(other_id, cached["payload"])
        return result

    def _record_forecast_warnings(self, pump_id: str, forecast: dict, risk: dict | None) -> None:
        """Keep forecast limit-crossing warnings in the issue history, off the request path."""
        if getattr(self, "issue_event_repository", None) is None:
            return
        snapshot, risk_snapshot = dict(forecast), dict(risk or {})

        def persist() -> None:
            try:
                self.issue_event_repository.reconcile_forecast_warnings(pump_id, snapshot, risk_snapshot)
                self.issue_event_store_error = None
            except Exception as exc:
                self.issue_event_store_error = str(exc)
                logger.exception("Could not persist forecast warnings for %s", pump_id)

        self.issue_event_executor.submit(persist)

    def _record_slow_trends(self, pump_id: str, maintenance: dict) -> None:
        """Keep slowly developing issues in the issue history, off the request path."""
        if getattr(self, "issue_event_repository", None) is None:
            return
        snapshot = dict(maintenance)

        def persist() -> None:
            try:
                self.issue_event_repository.reconcile_slow_trends(pump_id, snapshot)
                self.issue_event_store_error = None
            except Exception as exc:
                self.issue_event_store_error = str(exc)
                logger.exception("Could not persist slow-trend events for %s", pump_id)

        self.issue_event_executor.submit(persist)

    async def refresh_forecast(self, pump_id: str) -> dict:
        """Run the hourly long-horizon Timestral forecast, then cache it."""
        profile = self.profile_repository.get(pump_id)
        if not profile or profile.get("approval_status") != "APPROVED":
            return {"state": "CONFIGURATION_REQUIRED"}
        try:
            payload = await self._forecast_for(pump_id, profile)
        except Exception as exc:
            logger.exception("Scheduled forecast failed for %s", pump_id)
            payload = {"state": "MODEL_UNAVAILABLE", "detail": str(exc)}
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
            self.forecast_cache[pump_id] = {"created_at": datetime.now(timezone.utc), "payload": payload}
            return payload

        if payload.get("state") == "FORECAST_READY":
            self._record_forecast_warnings(
                pump_id, payload, forecast_risk(payload["prediction"], profile, payload["interval_minutes"]),
            )
        current = _in_memory_cache.get(pump_id)
        if current and current.get("state") == "NORMAL":
            risk_source = payload
            risk = forecast_risk(risk_source["prediction"], profile, risk_source["interval_minutes"]) if risk_source.get("state") == "FORECAST_READY" else None
            if risk is not None:
                risk["forecast_horizon_hours"] = risk_source["horizon_minutes"] // 60
            current["forecast"] = payload
            current["future_electrical_risk"] = risk
            current["model_status"] = payload.get("state", "UNKNOWN")
        return payload

    async def _forecast_schedule(self) -> None:
        """Refresh approved-pump forecasts without API traffic."""
        while self.is_running:
            async with self._historical_job_lock:
                try:
                    pump_ids = await self._approved_pump_ids()
                    for pump_id in pump_ids:
                        if not self.is_running:
                            break
                        try:
                            await self.refresh_forecast(pump_id)
                        except Exception:
                            logger.exception("Forecast job failed for %s; continuing fleet cycle", pump_id)
                finally:
                    self._first_forecast_cycle_done.set()
            await asyncio.sleep(settings.FORECAST_REFRESH_MINUTES * 60)

    async def _maintenance_schedule(self) -> None:
        """Refresh slow-degradation maintenance estimates separately from forecasts."""
        await self._first_forecast_cycle_done.wait()
        while self.is_running:
            async with self._historical_job_lock:
                pump_ids = await self._approved_pump_ids()
                for pump_id in pump_ids:
                    if not self.is_running:
                        break
                    try:
                        await self.refresh_electrical_maintenance(pump_id)
                    except Exception:
                        logger.exception("Maintenance job failed for %s; continuing fleet cycle", pump_id)
            await asyncio.sleep(settings.MAINTENANCE_REFRESH_MINUTES * 60)

    def _store_profile_result(self, pump_id: str, frame: pd.DataFrame, profile: dict | None, state: str, *, detail: str | None = None, data_quality: dict | None = None, data_age_seconds: float | None = None, faults: list[dict] | None = None, cycling: dict | None = None, forecast: dict | None = None, risk: dict | None = None, maintenance: dict | None = None, analysis_reading: pd.Series | None = None, early_warnings: list[dict] | None = None, live_session_rows: int | None = None) -> None:
        """Assemble and cache the complete dashboard state for one pump."""
        latest = frame.iloc[-1] if not frame.empty else pd.Series(dtype=object)
        timestamp = latest.get("_time") if not frame.empty else None
        timestamp = timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp
        if state == "STALE_DATA":
            detail = "No recent data. Live condition and guidance are unavailable."
        current_values = phase_payload(latest) if not frame.empty else {}
        faults = faults or []
        early_warnings = early_warnings or []
        # A lost phase also pushes other rules over their limits; show those as
        # supporting evidence of the one root cause instead of separate problems.
        present = {fault.get("code") for fault in faults}
        consequences = {
            "PHASE_LOSS": {"PHASE_CURRENT_IMBALANCE"},
            "VOLTAGE_PHASE_LOSS": {"UNDER_VOLTAGE", "VOLTAGE_IMBALANCE"},
        }

        def root_cause(code: str | None) -> str | None:
            return next((root for root, codes in consequences.items() if root in present and code in codes), None)

        faults = [
            {**fault, "role": "supporting", "supports": root_cause(fault.get("code"))}
            if root_cause(fault.get("code"))
            else {**fault, "role": "primary"}
            for fault in faults
        ]
        messages = [fault["evidence"] for fault in faults]
        if detail:
            messages.insert(0, detail)
        model_state = (forecast or {}).get("state", "RULES_ONLY")
        if risk:
            messages.extend(signal["evidence"] for signal in risk["signals"])
        health = calculate_smoothed_health(
            self.health_smoother, pump_id, timestamp, analysis_reading if analysis_reading is not None else latest,
            profile_health_baseline(profile), faults, [], early_warnings, cycling,
            None, state, data_quality,
        )
        self._in_memory_profile_payload = {
            "pump_id": pump_id, "pump_status": str(latest.get("pump_status") or "UNKNOWN").upper(), "state": state,
            "health": health, "health_score": health["score"],
            "anomaly_score": None, "smoothed_score": None, "threshold": 0.0, "current_streak": 0,
            "engineering_flags": [fault["code"] for fault in faults], "engineering_messages": messages,
            "detected_faults": faults, "baseline_deviations": [], "early_warnings": early_warnings, "short_cycling": cycling or {},
            "model_evidence": None, "informational_status": {}, "current_values": current_values,
            "baseline_ranges": {}, "data_quality": data_quality or {"status": "SKIPPED"}, "live_context_diagnostics": {},
            "data_quality_warnings": list(dict.fromkeys((data_quality or {}).get("unavailable_fields") or [])) if (data_quality or {}).get("status") == "WARN" else [],
            "baseline_issues": [], "state_reason": detail or (messages[0] if messages else state),
            "data_timestamp": timestamp, "analysis_context": "PROFILE_ENGINEERING", "last_on_timestamp": timestamp,
            "last_data_timestamp": timestamp, "data_age_seconds": data_age_seconds,
            "status_timestamp": timestamp, "last_processed_timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_approved": False, "model_version": "timestral-123m", "model_status": model_state,
            "comparison_window_rows": len(frame), "live_session_rows": live_session_rows, "baseline_source": None, "baseline_quality": None,
            "baseline_row_count": None, "baseline_target_row_count": None, "baseline_minimum_row_count": None,
            "baseline_session_count": None, "baseline_period": None, "baseline_statistics": None,
            "baseline_pending_detail": None, "engineering_profile": profile,
            "forecast": forecast or {"state": "RULES_ONLY"}, "future_electrical_risk": risk,
            "electrical_maintenance": None if state == "STALE_DATA" else maintenance or self._cached_maintenance(pump_id) or {"state": "MAINTENANCE_PENDING", "signals": []},
        }
        _in_memory_cache[pump_id] = self._in_memory_profile_payload
        self._record_issue_events(_in_memory_cache[pump_id])

    def _record_issue_events(self, payload: dict) -> None:
        """Persist active/resolved issue lifecycle changes off the request path."""
        if self.issue_event_repository is None:
            return
        snapshot = dict(payload)

        def persist() -> None:
            """Write one immutable pump snapshot to the issue-event repository."""
            try:
                self.issue_event_repository.reconcile(snapshot)
                self.issue_event_store_error = None
            except Exception as exc:
                self.issue_event_store_error = str(exc)
                logger.exception("Could not persist issue events for %s", snapshot.get("pump_id"))

        self.issue_event_executor.submit(persist)


    async def process_pump(self, pump_id: str):
        """Compatibility entry point that performs one live pump refresh."""
        await self.refresh_pump(pump_id)

    async def discover_pumps(self) -> list[str]:
        """Fetch pump IDs from Influx on demand; keep the last list if Influx fails."""
        from backend.engine import config
        try:
            found = await asyncio.to_thread(self.influx_service.get_all_pump_ids)
        except Exception as exc:
            logger.exception("Pump discovery failed")
            self.pump_discovery_error = str(exc) or type(exc).__name__
            return config.WORKING_PUMPS
        self.pump_discovery_error = None
        config.WORKING_PUMPS = list(dict.fromkeys(found))
        return config.WORKING_PUMPS

    async def start(self):
        """Start the forecast and maintenance schedules off the request path."""
        self.is_running = True
        self._first_forecast_cycle_done.clear()
        self.forecast_task = asyncio.create_task(self._forecast_schedule())
        self.maintenance_task = asyncio.create_task(self._maintenance_schedule())

    def stop(self):
        """Stop schedules and close worker-owned repositories and executors."""
        self.is_running = False
        if self.forecast_task is not None:
            self.forecast_task.cancel()
        if self.maintenance_task is not None:
            self.maintenance_task.cancel()
        if hasattr(self.profile_repository, "close"):
            self.profile_repository.close()
        self.issue_event_executor.shutdown(wait=True)
        if self.issue_event_repository is not None:
            self.issue_event_repository.close()


worker_instance = PredictionWorker()
