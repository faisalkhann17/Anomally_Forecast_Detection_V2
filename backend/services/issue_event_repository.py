"""MongoDB persistence for pump issue lifecycle events."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from backend.engine.readable_text import count


CRITICAL_FAULTS = {
    "PHASE_LOSS", "VOLTAGE_PHASE_LOSS", "DRY_RUN", "MOTOR_OVERLOAD",
}


def _utc_datetime(value: Any, fallback: datetime | None = None) -> datetime:
    """Normalize a timestamp-like value to timezone-aware UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif value:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = fallback or datetime.now(timezone.utc)
    else:
        parsed = fallback or datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def observed_issues(payload: dict) -> tuple[list[dict], set[str]]:
    """Project one inference result into active, client-readable issue events."""
    state = str(payload.get("state") or "UNKNOWN").upper()
    state = {
        "PROVISIONAL_NORMAL": "NORMAL",
        "PROVISIONAL_ANOMALY": "ANOMALY",
    }.get(state, state)
    issues: list[dict] = []
    resolve_types: set[str] = set()

    if state in {"NORMAL", "ANOMALY"}:
        resolve_types.update({"FAULT", "EARLY_WARNING", "SYSTEM"})
    elif state in {"DATA_ERROR", "DATA_QUALITY_ERROR", "MODEL_ERROR"}:
        resolve_types.add("SYSTEM")

    faults = payload.get("detected_faults") or []
    if state == "ANOMALY":
        if faults:
            primary = [fault for fault in faults if fault.get("role") != "supporting"] or faults
            for fault in primary:
                code = str(fault.get("code") or "UNKNOWN_FAULT").upper()
                issues.append({
                    "event_key": f"FAULT:{code}",
                    "event_type": "FAULT",
                    "code": code,
                    "label": fault.get("label") or code.replace("_", " ").title(),
                    "severity": "CRITICAL" if code in CRITICAL_FAULTS else "WARNING",
                    "evidence": fault.get("evidence"),
                })
        else:
            issues.append({
                "event_key": "FAULT:UNUSUAL_PUMP_BEHAVIOUR",
                "event_type": "FAULT",
                "code": "UNUSUAL_PUMP_BEHAVIOUR",
                "label": "Unusual Pump Behaviour",
                "severity": "WARNING",
                "evidence": payload.get("state_reason"),
            })

    if state == "NORMAL":
        for warning in payload.get("early_warnings") or []:
            code = str(warning.get("code") or warning.get("field") or "EARLY_WARNING").upper()
            issues.append({
                "event_key": f"EARLY_WARNING:{code}",
                "event_type": "EARLY_WARNING",
                "code": code,
                "label": warning.get("label") or warning.get("sensor") or code.replace("_", " ").title(),
                "severity": "EARLY_SIGN",
                "evidence": warning.get("trend_evidence") or warning.get("evidence"),
            })

    system_labels = {
        "DATA_QUALITY_ERROR": "Reading Problem",
        "DATA_ERROR": "Data Unavailable",
        "MODEL_ERROR": "Analysis Unavailable",
    }
    if state in system_labels:
        issues.append({
            "event_key": f"SYSTEM:{state}",
            "event_type": "SYSTEM",
            "code": state,
            "label": system_labels[state],
            "severity": "SYSTEM",
            "evidence": payload.get("state_reason"),
        })
    return issues, resolve_types


SLOW_TREND = "SLOW_TREND"


def observed_slow_trends(maintenance: dict) -> list[dict] | None:
    """Project a maintenance result into slow-trend issues; None when it cannot be judged.

    Returns None for failed or history-limited analyses so existing trend events
    are neither resolved nor changed by an incomplete run.
    """
    combined = (maintenance or {}).get("combined_slow_degradation")
    if not isinstance(combined, dict) or "developing_issues" not in combined:
        return None
    if str(maintenance.get("state") or "").upper() in {"MAINTENANCE_UNAVAILABLE", "INSUFFICIENT_HISTORY"}:
        return None
    issues = []
    for item in combined.get("developing_issues") or []:
        code = str(item.get("code") or "SLOW_TREND").upper()
        days = item.get("estimated_days_to_threshold")
        evidence = item.get("evidence") or ""
        if days is not None:
            evidence = f"{evidence} At this rate, the limit may be reached in about {count(max(0, round(float(days))), 'day')}.".strip()
        issues.append({
            "event_key": f"{SLOW_TREND}:{code}",
            "event_type": SLOW_TREND,
            "code": code,
            "label": item.get("label") or code.replace("_", " ").title(),
            "severity": "DEVELOPING",
            "evidence": evidence or None,
            "group": item.get("group_label") or item.get("group"),
        })
    guidance = combined.get("maintenance_guidance") or {}
    if combined.get("maintenance_required"):
        urgent = guidance.get("state") == "URGENT_INSPECTION"
        issues.append({
            "event_key": f"{SLOW_TREND}:MAINTENANCE_RECOMMENDED",
            "event_type": SLOW_TREND,
            "code": "MAINTENANCE_RECOMMENDED",
            "label": "Maintenance recommended" + (" (urgent)" if urgent else ""),
            "severity": "MAINTENANCE",
            "evidence": guidance.get("reason") or "Issues are developing in two or more independent areas.",
            "group": ", ".join(guidance.get("issue_group_labels") or []) or None,
        })
    return issues


FORECAST_WARNING = "FORECAST_WARNING"
FORECAST_LABELS = {
    "RISING_LOAD_TREND": "Forecast: motor current may exceed its limit",
    "VOLTAGE_STABILITY_DEGRADING": "Forecast: voltage may fall below its limit",
    "OVER_VOLTAGE_TREND": "Forecast: voltage may rise above its limit",
    "POWER_FACTOR_DEGRADING": "Forecast: power factor may fall below its limit",
    "PHASE_DIVERGENCE_TREND": "Forecast: phase currents may become unbalanced",
    "ENERGY_INTENSITY_DRIFT": "Forecast: power may exceed its limit",
}


def observed_forecast_warnings(forecast: dict, risk: dict | None) -> list[dict] | None:
    """Project one forecast run into warning issues; None when the run cannot be judged.

    Exact "crosses in N hours" values are not stored: they were found to run early,
    so the evidence names the forecast window instead.
    """
    if (forecast or {}).get("state") != "FORECAST_READY" or not risk or risk.get("state") != "READY":
        return None
    if risk.get("attention_level") != "MAINTENANCE_ATTENTION":
        return []
    hours = int((forecast.get("horizon_minutes") or 0) // 60)
    window = f"within the next {count(max(1, round(hours / 24)), 'day')}" if hours else "within the forecast period"
    issues = []
    for signal in risk.get("signals") or []:
        code = str(signal.get("code") or "FORECAST_SIGNAL").upper()
        issues.append({
            "event_key": f"{FORECAST_WARNING}:{code}",
            "event_type": FORECAST_WARNING,
            "code": code,
            "label": FORECAST_LABELS.get(code, "Forecast: " + code.replace("_", " ").lower()),
            "severity": "FORECAST",
            "evidence": f"{signal.get('evidence') or 'The forecast reaches a safe limit.'} Possible {window}.",
        })
    return issues


class MongoIssueEventRepository:
    """One document per issue occurrence; active events are updated in place."""

    def __init__(self, uri: str, database: str, collection: str, timeout_ms: int):
        """Connect to local MongoDB and create issue lookup indexes."""
        from pymongo import ASCENDING, DESCENDING, MongoClient
        from pymongo.uri_parser import parse_uri

        if not uri:
            raise ValueError("MONGODB_URI is required for issue event persistence.")
        hosts = {host for host, _ in parse_uri(uri)["nodelist"]}
        if not hosts or not hosts.issubset({"127.0.0.1", "localhost", "::1"}):
            raise ValueError("Issue event persistence requires a local MongoDB URI.")
        self.client = MongoClient(
            uri,
            appname="pumpwatch-issue-history",
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
        )
        self.collection = self.client[database][collection]
        self.client.admin.command("ping")
        self.collection.create_index(
            [("active_key", ASCENDING)], unique=True, sparse=True,
            name="unique_active_issue",
        )
        self.collection.create_index(
            [("pump_id", ASCENDING), ("started_at", DESCENDING)],
            name="pump_started_desc",
        )
        self.collection.create_index(
            [("pump_id", ASCENDING), ("status", ASCENDING), ("event_type", ASCENDING)],
            name="pump_status_type",
        )

    def reconcile(self, payload: dict) -> None:
        """Upsert observed issues and resolve active issues no longer present."""
        pump_id = str(payload.get("pump_id") or "").strip()
        if not pump_id:
            return
        now = datetime.now(timezone.utc)
        event_time = _utc_datetime(payload.get("data_timestamp"), now)
        issues, resolve_types = observed_issues(payload)
        active_keys = []

        health = payload.get("health") or {}
        for issue in issues:
            event_key = issue["event_key"]
            active_key = f"{pump_id}:{event_key}"
            active_keys.append(active_key)
            self.collection.update_one(
                {"active_key": active_key},
                {
                    "$set": {
                        "pump_id": pump_id,
                        "active_key": active_key,
                        "event_key": event_key,
                        "event_type": issue["event_type"],
                        "code": issue["code"],
                        "label": issue["label"],
                        "severity": issue["severity"],
                        "status": "ACTIVE",
                        "latest_evidence": issue.get("evidence"),
                        "health_score": health.get("score"),
                        "health_band": health.get("band"),
                        "updated_at": now,
                    },
                    "$setOnInsert": {
                        "started_at": event_time,
                        "created_at": now,
                        "schema_version": 1,
                    },
                    "$unset": {
                        "last_seen_at": "",
                        "last_observed_at": "",
                    },
                },
                upsert=True,
            )

        if resolve_types:
            query: dict[str, Any] = {
                "pump_id": pump_id,
                "status": "ACTIVE",
                "event_type": {"$in": sorted(resolve_types)},
            }
            if active_keys:
                query["active_key"] = {"$nin": active_keys}
            self.collection.update_many(
                query,
                {
                    "$set": {
                        "status": "RESOLVED",
                        "resolved_at": event_time,
                        "updated_at": now,
                    },
                    "$unset": {
                        "active_key": "",
                        "last_seen_at": "",
                        "last_observed_at": "",
                    },
                },
            )

    def reconcile_slow_trends(self, pump_id: str, maintenance: dict) -> int | None:
        """Open, update and resolve SLOW_TREND events from one maintenance assessment.

        Returns how many trend issues are active, or None when the assessment was
        skipped because it could not be judged.
        """
        pump_id = str(pump_id or "").strip()
        issues = observed_slow_trends(maintenance)
        if not pump_id or issues is None:
            return None
        now = datetime.now(timezone.utc)
        event_time = _utc_datetime(maintenance.get("generated_at"), now)
        active_keys = []
        for issue in issues:
            active_key = f"{pump_id}:{issue['event_key']}"
            active_keys.append(active_key)
            self.collection.update_one(
                {"active_key": active_key},
                {
                    "$set": {
                        "pump_id": pump_id, "active_key": active_key, "event_key": issue["event_key"],
                        "event_type": SLOW_TREND, "code": issue["code"], "label": issue["label"],
                        "severity": issue["severity"], "status": "ACTIVE", "latest_evidence": issue.get("evidence"),
                        "trend_group": issue.get("group"), "updated_at": now,
                    },
                    "$setOnInsert": {"started_at": event_time, "created_at": now, "schema_version": 1},
                },
                upsert=True,
            )
        query: dict[str, Any] = {"pump_id": pump_id, "status": "ACTIVE", "event_type": SLOW_TREND}
        if active_keys:
            query["active_key"] = {"$nin": active_keys}
        self.collection.update_many(
            query,
            {"$set": {"status": "RESOLVED", "resolved_at": event_time, "updated_at": now}, "$unset": {"active_key": ""}},
        )
        return len(issues)

    def reconcile_forecast_warnings(self, pump_id: str, forecast: dict, risk: dict | None) -> int | None:
        """Open, update and resolve FORECAST_WARNING events from one forecast run."""
        pump_id = str(pump_id or "").strip()
        issues = observed_forecast_warnings(forecast, risk)
        if not pump_id or issues is None:
            return None
        now = datetime.now(timezone.utc)
        event_time = _utc_datetime(forecast.get("generated_at"), now)
        active_keys = []
        for issue in issues:
            active_key = f"{pump_id}:{issue['event_key']}"
            active_keys.append(active_key)
            self.collection.update_one(
                {"active_key": active_key},
                {
                    "$set": {
                        "pump_id": pump_id, "active_key": active_key, "event_key": issue["event_key"],
                        "event_type": FORECAST_WARNING, "code": issue["code"], "label": issue["label"],
                        "severity": issue["severity"], "status": "ACTIVE", "latest_evidence": issue["evidence"],
                        "updated_at": now,
                    },
                    "$setOnInsert": {"started_at": event_time, "created_at": now, "schema_version": 1},
                },
                upsert=True,
            )
        query: dict[str, Any] = {"pump_id": pump_id, "status": "ACTIVE", "event_type": FORECAST_WARNING}
        if active_keys:
            query["active_key"] = {"$nin": active_keys}
        self.collection.update_many(
            query,
            {"$set": {"status": "RESOLVED", "resolved_at": event_time, "updated_at": now}, "$unset": {"active_key": ""}},
        )
        return len(issues)

    def list_events(
        self,
        pump_id: str,
        *,
        status: str | None = None,
        event_type: str | None = None,
        code: str | None = None,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 50,
    ) -> dict:
        """Query pump issue history using optional filters."""
        query: dict[str, Any] = {"pump_id": pump_id}
        if status and status != "ALL":
            query["status"] = status
        if event_type and event_type != "ALL":
            query["event_type"] = event_type
        if code:
            query["code"] = code
        if from_time or to_time:
            time_query = {}
            if from_time:
                time_query["$gte"] = _utc_datetime(from_time)
            if to_time:
                time_query["$lte"] = _utc_datetime(to_time)
            query["started_at"] = time_query

        total = self.collection.count_documents(query)
        cursor = self.collection.find(query).sort("started_at", -1).limit(limit)
        events = []
        for document in cursor:
            events.append({
                "id": str(document.get("_id")),
                "pump_id": document.get("pump_id"),
                "event_type": document.get("event_type"),
                "code": document.get("code"),
                "label": document.get("label"),
                "severity": document.get("severity"),
                "status": document.get("status"),
                "started_at": _iso(document.get("started_at")),
                "resolved_at": _iso(document.get("resolved_at")),
                "evidence": document.get("latest_evidence"),
                "health_score": document.get("health_score"),
                "health_band": document.get("health_band"),
            })
        return {"pump_id": pump_id, "total": total, "events": events}

    def close(self) -> None:
        """Close the MongoDB client owned by this repository."""
        self.client.close()


def _iso(value: Any) -> str | None:
    """Serialize a timestamp as UTC ISO text for API responses."""
    return _utc_datetime(value).isoformat() if value else None
