"""Persistent, versioned engineering profiles keyed by pump ID.

The running service stores profiles in MongoDB (``MongoPumpProfileRepository``).
``PumpProfileRepository`` is a file-backed equivalent kept for tests and
offline tools; the API does not use it.
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from backend.config import settings


logger = logging.getLogger(__name__)

# Fields compared when recording what changed between two profile versions.
HISTORY_TRACKED_FIELDS = (
    "rated_power_kw", "rated_voltage_v", "rated_current_a", "rated_frequency_hz",
    "phase_count", "voltage_type", "approval_status", "approved_by",
)


def _tracked_values(profile: dict | None) -> dict:
    """Flatten the fields whose changes are shown in profile history."""
    profile = profile or {}
    values = {field: profile.get(field) for field in HISTORY_TRACKED_FIELDS if field in profile}
    for key, value in (profile.get("operating_limits") or {}).items():
        values[f"operating_limits.{key}"] = value
    return values


def profile_changes(previous: dict | None, current: dict) -> list[dict]:
    """List each tracked field whose value differs between two versions."""
    before, after = _tracked_values(previous), _tracked_values(current)
    return [
        {"field": field, "before": before.get(field), "after": after.get(field)}
        for field in sorted(set(before) | set(after))
        if before.get(field) != after.get(field)
    ]


def history_entry(pump_id: str, previous: dict | None, saved: dict, action: str, identity: str) -> dict:
    """Build one immutable history record: who, when, what changed, full snapshot."""
    return {
        "pump_id": pump_id,
        "profile_version": saved["version"],
        "previous_version": (previous or {}).get("version"),
        "action": action,
        "identity": identity,
        "recorded_at": datetime.now(timezone.utc),
        "changes": profile_changes(previous, saved),
        "profile": copy.deepcopy(saved),
    }


def _public_history(entry: dict) -> dict:
    """Serialize a history record for API responses."""
    result = {key: value for key, value in entry.items() if key != "_id"}
    recorded = result.get("recorded_at")
    if isinstance(recorded, datetime):
        result["recorded_at"] = (recorded if recorded.tzinfo else recorded.replace(tzinfo=timezone.utc)).isoformat()
    return result


class PumpProfileRepository:
    """File-backed profile store for tests and offline tools (not used by the API)."""
    def __init__(self, path: Path) -> None:
        """Configure and create the local profile-store directory."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict] = []

    def load_all(self) -> dict[str, dict]:
        """Load every saved profile, returning empty data on invalid JSON."""
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def get(self, pump_id: str) -> dict | None:
        """Return the latest engineering profile for one pump."""
        return self.load_all().get(pump_id)

    def upsert(self, pump_id: str, profile: dict, *, action: str = "UPSERT", identity: str = "system") -> dict:
        """Increment profile version, save it and keep an in-memory history entry."""
        profiles = self.load_all()
        previous = profiles.get(pump_id) or {}
        version = int(previous.get("version", 0)) + 1
        saved = {**profile, "pump_id": pump_id, "version": version,
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        profiles[pump_id] = saved
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        self._history.append(history_entry(pump_id, previous or None, saved, action, identity))
        return saved

    def list_history(self, pump_id: str, limit: int = 50) -> list[dict]:
        """Return this pump's history, newest version first."""
        entries = [entry for entry in self._history if entry["pump_id"] == pump_id]
        entries.sort(key=lambda entry: entry["profile_version"], reverse=True)
        return [_public_history(entry) for entry in entries[:limit]]


class ProfileStoreUnavailable(RuntimeError):
    """MongoDB could not be reached, so profiles cannot be read or saved."""


class ProfileUpdateConflict(RuntimeError):
    """Another request changed the same profile while this one was saving."""


class MongoPumpProfileRepository:
    """Store versioned engineering profiles directly in local MongoDB.

    MongoDB is the only source of truth. Reads keep the last successfully loaded
    profiles in memory so live monitoring keeps its approved limits during a
    short MongoDB outage; writes always require MongoDB and fail loudly.
    """

    _MAX_SAVE_ATTEMPTS = 5

    def __init__(self, uri: str, database: str, collection: str, timeout_ms: int,
                 *, history_collection: str = "pumpwatch_profile_history") -> None:
        """Validate the local MongoDB URI and name the profile and history collections."""
        parsed = urlsplit(uri)
        if parsed.scheme != "mongodb" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or "," in parsed.netloc:
            raise ValueError("Profile MongoDB URI must point to a single local loopback host.")
        self._uri = uri
        self._database = database
        self._collection_name = collection
        self._history_collection_name = history_collection
        self._history_indexed = False
        self.history_error: str | None = None
        self._timeout_ms = timeout_ms
        self._client = None
        self._last_known: dict[str, dict] = {}
        self._loaded_all = False
        self.last_error: str | None = None

    def _collection(self):
        """Lazily connect to the configured MongoDB profile collection."""
        if self._client is None:
            from pymongo import MongoClient

            self._client = MongoClient(
                self._uri, appname="pumpwatch-profile-store",
                serverSelectionTimeoutMS=self._timeout_ms,
                connectTimeoutMS=self._timeout_ms,
            )
        return self._client[self._database][self._collection_name]

    def _history_collection(self):
        """Return the history collection, creating its lookup index once."""
        collection = self._collection().database[self._history_collection_name]
        if not self._history_indexed:
            from pymongo import ASCENDING, DESCENDING

            collection.create_index([("pump_id", ASCENDING), ("profile_version", DESCENDING)], name="pump_version_desc")
            self._history_indexed = True
        return collection

    def _record_history(self, pump_id: str, previous: dict | None, saved: dict, action: str, identity: str) -> None:
        """Write one history entry; a failure is logged and repaired by backfill_history."""
        from pymongo.errors import PyMongoError

        entry = {"_id": f"{pump_id}:{saved['version']}", **history_entry(pump_id, previous, saved, action, identity)}
        try:
            self._history_collection().update_one({"_id": entry["_id"]}, {"$setOnInsert": entry}, upsert=True)
            self.history_error = None
        except PyMongoError as exc:
            self.history_error = f"{type(exc).__name__}: {exc}"
            logger.error("Profile saved but its history entry was not written for %s v%s: %s",
                         pump_id, saved["version"], self.history_error)

    def list_history(self, pump_id: str, limit: int = 50) -> list[dict]:
        """Return one pump's profile history, newest version first."""
        from pymongo.errors import PyMongoError

        try:
            cursor = self._history_collection().find({"pump_id": pump_id}).sort("profile_version", -1).limit(int(limit))
            return [_public_history(entry) for entry in cursor]
        except PyMongoError as exc:
            raise self._unavailable(exc, f"reading history for pump {pump_id}") from exc

    def backfill_history(self) -> int:
        """Record each profile's current version when history has no entry for it.

        Covers profiles created before history existed (IMPORTED) and saves whose
        history write failed (RECOVERED, compared with the previous entry).
        """
        history = self._history_collection()
        written = 0
        for document in self._collection().find({}):
            profile = self._clean(document)
            pump_id, version = str(document["_id"]), profile.get("version")
            if version is None or history.find_one({"_id": f"{pump_id}:{version}"}, {"_id": 1}):
                continue
            previous_entry = history.find_one(
                {"pump_id": pump_id, "profile_version": {"$lt": version}}, sort=[("profile_version", -1)],
            )
            previous = (previous_entry or {}).get("profile")
            approved = profile.get("approval_status") == "APPROVED"
            identity = (profile.get("approved_by") if approved else profile.get("modified_by")) or "unknown"
            entry = {"_id": f"{pump_id}:{version}", **history_entry(
                pump_id, previous, profile, "RECOVERED" if previous else "IMPORTED", identity,
            )}
            if history.update_one({"_id": entry["_id"]}, {"$setOnInsert": entry}, upsert=True).upserted_id is not None:
                written += 1
        self.history_error = None
        return written

    @staticmethod
    def _clean(document: dict) -> dict:
        """Return the API profile record without MongoDB's internal key."""
        return {key: value for key, value in document.items() if key != "_id"}

    def _unavailable(self, exc: Exception, action: str) -> ProfileStoreUnavailable:
        """Record a MongoDB failure and build the error raised to callers."""
        self.last_error = f"{type(exc).__name__}: {exc}"
        logger.error("MongoDB profile store unavailable while %s: %s", action, self.last_error)
        return ProfileStoreUnavailable("Engineering profile store (MongoDB) is unavailable.")

    def load_all(self) -> dict[str, dict]:
        """Return every profile, or the last loaded set while MongoDB is down."""
        from pymongo.errors import PyMongoError

        try:
            profiles = {str(document["_id"]): self._clean(document) for document in self._collection().find({})}
        except PyMongoError as exc:
            error = self._unavailable(exc, "loading all profiles")
            if self._loaded_all:
                return copy.deepcopy(self._last_known)
            raise error from exc
        self._last_known = profiles
        self._loaded_all = True
        self.last_error = None
        return copy.deepcopy(profiles)

    def get(self, pump_id: str) -> dict | None:
        """Return one profile, or its last loaded copy while MongoDB is down."""
        from pymongo.errors import PyMongoError

        try:
            document = self._collection().find_one({"_id": pump_id})
        except PyMongoError as exc:
            error = self._unavailable(exc, f"reading pump {pump_id}")
            if self._loaded_all or pump_id in self._last_known:
                return copy.deepcopy(self._last_known.get(pump_id))
            raise error from exc
        self.last_error = None
        if document is None:
            self._last_known.pop(pump_id, None)
            return None
        profile = self._clean(document)
        self._last_known[pump_id] = profile
        return copy.deepcopy(profile)

    def upsert(self, pump_id: str, profile: dict, *, action: str = "UPSERT", identity: str = "system") -> dict:
        """Save a new profile version and its history; concurrent edits never overwrite."""
        from pymongo.errors import DuplicateKeyError, PyMongoError

        try:
            collection = self._collection()
            for _ in range(self._MAX_SAVE_ATTEMPTS):
                current = collection.find_one({"_id": pump_id})
                version = int((current or {}).get("version", 0)) + 1
                saved = {
                    **{key: value for key, value in profile.items() if key != "_id"},
                    "pump_id": pump_id, "version": version,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                document = {"_id": pump_id, **saved}
                if current is None:
                    try:
                        collection.insert_one(document)
                    except DuplicateKeyError:
                        continue
                elif collection.replace_one({"_id": pump_id, "version": current.get("version")}, document).matched_count == 0:
                    continue
                self._last_known[pump_id] = copy.deepcopy(saved)
                self.last_error = None
                self._record_history(pump_id, self._clean(current) if current else None, saved, action, identity)
                return saved
        except PyMongoError as exc:
            raise self._unavailable(exc, f"saving pump {pump_id}") from exc
        raise ProfileUpdateConflict(f"Profile for {pump_id} was changed by another request; reload and try again.")

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()


def create_profile_repository(_legacy_path: Path | None = None) -> MongoPumpProfileRepository:
    """Build the MongoDB profile store (the optional path argument is ignored)."""
    return MongoPumpProfileRepository(
        settings.PROFILE_MONGODB_URI, settings.PROFILE_MONGODB_DATABASE,
        settings.MONGODB_PROFILE_COLLECTION, settings.PROFILE_MONGODB_TIMEOUT_MS,
        history_collection=settings.MONGODB_PROFILE_HISTORY_COLLECTION,
    )
