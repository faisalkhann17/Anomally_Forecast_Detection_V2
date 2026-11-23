"""Persistent frozen electrical-maintenance baselines keyed by pump profile."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class MaintenanceBaselineRepository:
    """Store immutable maintenance baselines by pump and profile version."""
    def __init__(self, path: Path) -> None:
        """Configure the local JSON baseline store path."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, pump_id: str, profile_version: int) -> dict | None:
        """Return a baseline only when it matches the active profile version."""
        try:
            values = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except json.JSONDecodeError:
            values = {}
        record = values.get(pump_id) if isinstance(values, dict) else None
        return record if isinstance(record, dict) and record.get("profile_version") == profile_version else None

    def save(self, pump_id: str, profile_version: int, baseline: dict) -> dict:
        """Atomically persist a newly created profile-version baseline."""
        try:
            values = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except json.JSONDecodeError:
            values = {}
        saved = {**baseline, "profile_version": profile_version, "created_at": datetime.now(timezone.utc).isoformat()}
        values[pump_id] = saved
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        return saved
