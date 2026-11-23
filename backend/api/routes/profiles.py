"""Engineer-owned pump profile approval endpoints."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.auth import require_engineer
from backend.config import settings
from backend.engine.electrical_profile import profile_limit_warnings, validate_profile
from backend.engine.flow_suggestion import suggest_minimum_flow
from backend.services.pump_profile_repository import ProfileStoreUnavailable, ProfileUpdateConflict
from backend.workers.prediction_worker import worker_instance

router = APIRouter(prefix="/api/pumps", tags=["Pump Engineering Profiles"])
PROFILE_INPUT_FIELDS = (
    "rated_power_kw", "rated_voltage_v", "rated_current_a", "rated_frequency_hz",
    "phase_count",
)
OPERATING_LIMIT_FIELDS = (
    "current_min_a", "current_max_a", "voltage_min_v", "voltage_max_v",
    "power_factor_min", "total_kw_min", "total_kw_max",
    "current_imbalance_max_percent", "voltage_imbalance_max_percent",
    "flowrate_min",
)


def _profile_status(profile: dict | None) -> str:
    """Classify a stored profile as approved, draft or not configured."""
    if not profile:
        return "NOT_CONFIGURED"
    return "APPROVED" if profile.get("approval_status") == "APPROVED" else "DRAFT"


def build_profile_overview(influx_pump_ids: list[str], profiles: dict[str, dict], influx_error: str | None) -> dict:
    """Match Influx pump IDs with stored engineering profiles and count each group."""
    def entry(pump_id: str, has_telemetry: bool) -> dict:
        profile = profiles.get(pump_id) or {}
        return {
            "pump_id": pump_id,
            "profile_status": _profile_status(profile),
            "has_telemetry": has_telemetry,
            "version": profile.get("version"),
            "updated_at": profile.get("updated_at"),
            "approved_by": profile.get("approved_by"),
            "approved_at": profile.get("approved_at"),
        }

    known = list(dict.fromkeys(influx_pump_ids))
    pumps = [entry(pump_id, True) for pump_id in known]
    no_telemetry = [entry(pump_id, False) for pump_id in sorted(set(profiles) - set(known))]
    counts = {status: sum(item["profile_status"] == status for item in pumps)
              for status in ("APPROVED", "DRAFT", "NOT_CONFIGURED")}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "influx_available": influx_error is None,
        "influx_error": influx_error,
        "summary": {
            "total": len(pumps),
            "approved": counts["APPROVED"],
            "draft": counts["DRAFT"],
            "not_configured": counts["NOT_CONFIGURED"],
            "profiles_without_telemetry": len(no_telemetry),
        },
        "pumps": pumps,
        "profiles_without_telemetry": no_telemetry,
    }


async def _profile_store(method: str, *args, **kwargs):
    """Run a MongoDB profile-store call off the event loop with clear HTTP errors."""
    try:
        return await asyncio.to_thread(getattr(worker_instance.profile_repository, method), *args, **kwargs)
    except ProfileStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"{exc} The profile was not changed; try again when MongoDB is reachable.") from exc
    except ProfileUpdateConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/profile-status")
async def get_profile_overview(identity: str = Depends(require_engineer)):
    """List Influx pumps with their limit status: approved, draft or not configured."""
    pump_ids = await worker_instance.discover_pumps()
    profiles = await _profile_store("load_all")
    return build_profile_overview(pump_ids, profiles, getattr(worker_instance, "pump_discovery_error", None))


@router.get("/{pump_id}/profile/history")
async def get_profile_history(pump_id: str, limit: int = Query(50, ge=1, le=500), identity: str = Depends(require_engineer)):
    """Return every recorded profile version: who, when, what changed, full snapshot."""
    entries = await _profile_store("list_history", pump_id, limit)
    return {"pump_id": pump_id, "entries": entries}


FLOW_SUGGESTION_WINDOW_MINUTES = 5


@router.get("/{pump_id}/profile/flow-suggestion")
async def get_flow_suggestion(pump_id: str, identity: str = Depends(require_engineer)):
    """Suggest the minimum delivery flow from the pump's own running history (read-only)."""
    result: dict | None = None
    for days in settings.FLOW_SUGGESTION_LOOKBACK_DAYS:
        try:
            flows = await asyncio.to_thread(
                worker_instance.influx_service.fetch_running_flow, pump_id, days, FLOW_SUGGESTION_WINDOW_MINUTES,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Could not read the flow history from the database. Try again later.") from exc
        result = suggest_minimum_flow(
            flows,
            window_minutes=FLOW_SUGGESTION_WINDOW_MINUTES,
            margin_percent=settings.FLOW_SUGGESTION_MARGIN_PERCENT,
            minimum_running_hours=settings.FLOW_SUGGESTION_MIN_RUNNING_HOURS,
            lookback_days=days,
            outlier_warning_percent=settings.FLOW_SUGGESTION_OUTLIER_WARNING_PERCENT,
        )
        if result["state"] == "READY":
            break
    if result is None:
        raise HTTPException(status_code=503, detail="Flow suggestion is not configured.")
    return {"pump_id": pump_id, **result}


@router.get("/{pump_id}/profile")
async def get_profile(pump_id: str, identity: str = Depends(require_engineer)):
    """Return the complete authenticated engineering profile record."""
    profile = await _profile_store("get", pump_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="No engineering profile exists for this pump.")
    return profile


def limit_warnings(profile: dict) -> list[dict]:
    """Nameplate-relative sanity warnings using the configured tolerances."""
    return profile_limit_warnings(
        profile,
        voltage_tolerance_percent=settings.PROFILE_VOLTAGE_TOLERANCE_PERCENT,
        max_to_rated_ratio=settings.PROFILE_MAX_TO_RATED_RATIO,
        min_max_to_rated_ratio=settings.PROFILE_MIN_MAX_TO_RATED_RATIO,
    )


@router.put("/{pump_id}/profile")
async def save_profile(
    pump_id: str,
    profile: dict,
    acknowledge_warnings: bool = Query(False),
    identity: str = Depends(require_engineer),
):
    """Save a draft; any edit withdraws approval and invalidates old cached results.

    Impossible values are rejected outright. Limits that look inconsistent with
    the nameplate are returned as warnings and saved only once the engineer
    confirms them with ``acknowledge_warnings=true``.
    """
    if profile.get("pump_id") not in (None, pump_id):
        raise HTTPException(status_code=422, detail="pump_id must match the URL.")
    try:
        validated = validate_profile(profile)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    clean_profile = {field: validated[field] for field in PROFILE_INPUT_FIELDS if field in validated}
    clean_profile["operating_limits"] = {
        field: validated["operating_limits"][field]
        for field in OPERATING_LIMIT_FIELDS if field in validated["operating_limits"]
    }
    warnings = limit_warnings(clean_profile)
    if warnings and not acknowledge_warnings:
        raise HTTPException(status_code=422, detail={
            "message": "Some limits look inconsistent with the rated values. Check them, then confirm to save.",
            "requires_acknowledgement": True,
            "warnings": warnings,
        })
    now = datetime.now(timezone.utc).isoformat()
    saved = await _profile_store("upsert", pump_id, {
        **clean_profile, "approval_status": "DRAFT", "modified_by": identity, "modified_at": now,
        "acknowledged_warnings": [
            {**warning, "acknowledged_by": identity, "acknowledged_at": now} for warning in warnings
        ],
    }, action="PUT", identity=identity)
    worker_instance.invalidate_forecast(pump_id)
    return saved


@router.post("/{pump_id}/profile/approve")
async def approve_profile(pump_id: str, approval: dict | None = None, identity: str = Depends(require_engineer)):
    """Approve limits under the authenticated engineer identity, not a body claim."""
    profile = await _profile_store("get", pump_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Create the engineering profile before approving it.")
    if approval is not None and "approved_by" in approval and str(approval["approved_by"] or "").strip() != identity:
        raise HTTPException(status_code=422, detail="approved_by must match the authenticated identity.")
    if (profile.get("operating_limits") or {}).get("flowrate_min") is None:
        raise HTTPException(status_code=422, detail="Enter the minimum delivery flow and save the draft before approving.")
    saved = await _profile_store("upsert", pump_id, {
        **profile, "approval_status": "APPROVED", "approved_by": identity,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_version": int(profile.get("version", 0)) + 1,
    }, action="APPROVE", identity=identity)
    worker_instance.invalidate_forecast(pump_id)
    return saved
