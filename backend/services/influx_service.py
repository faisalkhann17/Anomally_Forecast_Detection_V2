"""Read pump telemetry from Influx and prepare recent or forecast time series."""

import logging
import json
import math
from datetime import datetime, timedelta, timezone
from threading import Lock
import pandas as pd
from typing import Optional
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from backend.config import settings
from backend.engine.data_quality import add_average_features
from backend.engine.electrical_profile import HYPHEN_COLUMN, mark_hyphen_rows as _mark_hyphen_rows, mark_nonfinite_rows
from backend.engine.data_quality import forecast_input_firewall
from backend.engine.on_only_aggregation import aggregate_on_only

logger = logging.getLogger("backend.services.influx_service")
LIVE_LOOKBACK_WINDOWS = (180, 1440, 10080)
STATUS_LOOKBACK_WINDOWS = (60, 1440, 10080)
LOOKBACK_PROBE_MARGIN = 10

# These channels must be checked explicitly before a row can enter forecasting.
REQUIRED_SENSOR_FIELDS: tuple[str, ...] = (
    "current_r", "current_y", "current_b",
    "voltage_r", "voltage_y", "voltage_b",
    "pf_r", "pf_y", "pf_b",
    "total_kw", "flowrate",
)
HOURLY_FORECAST_FIELDS: tuple[str, ...] = (
    "current_r", "current_y", "current_b",
    "voltage_r", "voltage_y", "voltage_b",
    "pf_r", "pf_y", "pf_b", "total_kw",
)
HOURLY_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "_pump_on", "on_fraction", "on_sample_count", "status_on_samples",
    "usable_on_samples", "invalid_on_samples", "no_draw_samples",
)
# pump_status is also required but comes as a tag, not a field — it's
# always present after pivot because it's part of the index columns.


class InfluxService:
    """Query configured InfluxDB telemetry and prepare analysis dataframes."""
    def __init__(self):
        """Load Influx connection metadata and initialize diagnostics state."""
        self.url = settings.INFLUXDB_URL
        self.token = settings.INFLUXDB_TOKEN
        self.org = settings.INFLUXDB_ORG
        self.bucket = settings.INFLUXDB_BUCKET
        self.measurement = settings.INFLUXDB_MEASUREMENT
        self.hyphen_fields = {}
        self.live_lookback_minutes = LIVE_LOOKBACK_WINDOWS
        self.status_lookback_minutes = STATUS_LOOKBACK_WINDOWS
        self._lookback_lock = Lock()
        self._lookback_diagnostics: dict[str, dict] = {}

    def _record_lookback(self, pump_id: str, name: str, window: int | None, fallback: bool) -> None:
        """Record which adaptive history window served one pump query."""
        with self._lookback_lock:
            item = self._lookback_diagnostics.setdefault(pump_id, {})
            item[f"{name}_window_minutes"] = window
            item[f"{name}_fallback_used"] = fallback

    def get_lookback_diagnostics(self, pump_id: str) -> dict:
        """Return a copy of adaptive lookback diagnostics for administration."""
        with self._lookback_lock:
            return dict(self._lookback_diagnostics.get(pump_id) or {})

    @staticmethod
    def _flux_string(value: str) -> str:
        """Encode a value safely for insertion into a Flux string literal."""
        return json.dumps(str(value))

    def _get_client(self) -> InfluxDBClient:
        """Create a validated short-lived InfluxDB client."""
        if not self.url:
            raise ValueError("INFLUXDB_URL is not configured.")
        if not self.token:
            raise ValueError("INFLUXDB_TOKEN is not configured.")
        return InfluxDBClient(url=self.url, token=self.token, org=self.org, timeout=60000)

    def get_all_pump_ids(self) -> list:
        """Return pumps that still have real telemetry rows in the measurement.

        Influx's schema tag-value index can retain product IDs after their data
        has been deleted. Selecting the latest required sensor row per pump
        keeps deleted/tombstoned series out of the operator pump selector.
        """
        query = f'''
        from(bucket: {self._flux_string(self.bucket)})
          |> range(start: 0)
          |> filter(fn: (r) => r._measurement == {self._flux_string(self.measurement)})
          |> filter(fn: (r) => r._field == "current_r")
          |> filter(fn: (r) => exists r.product_id)
          |> group(columns: ["product_id"])
          |> last()
          |> keep(columns: ["product_id"])
          |> group()
          |> sort(columns: ["product_id"])
        '''
        client = self._get_client()
        try:
            query_api = client.query_api()
            result = query_api.query(query=query)
            pump_ids = []
            for table in result:
                for record in table.records:
                    pump_id = record.values.get("product_id")
                    if pump_id:
                        pump_ids.append(str(pump_id))
            return sorted(set(pump_ids))
        finally:
            client.close()

    def fetch_raw_data(self, pump_id: Optional[str] = None, start: str = "-7d", stop: str = "now()", tail_count: Optional[int] = None, on_only: bool = False) -> pd.DataFrame:
        """
        Fetches long-format data from InfluxDB using a Flux query strictly on the configured INFLUXDB_MEASUREMENT.
        If pump_id is provided, filters by product_id tag.
        """
        query = f'''
        from(bucket: {self._flux_string(self.bucket)})
          |> range(start: {start}, stop: {stop})
          |> filter(fn: (r) => r._measurement == {self._flux_string(self.measurement)})
        '''
        if pump_id:
            query += f'  |> filter(fn: (r) => r.product_id == {self._flux_string(pump_id)})\n'

        if on_only:
            query += '  |> filter(fn: (r) => r.pump_status == "on" or r.pump_status == "ON" or r.pump_status == "running" or r.pump_status == "RUNNING")\n'

        if tail_count and not on_only:
            query += f'  |> tail(n: {tail_count})\n'

        logger.info(f"Querying InfluxDB for {pump_id or 'all pumps'} strictly on measurement '{self.measurement}' from {start} to {stop}")
        
        client = self._get_client()
        try:
            query_api = client.query_api()
            tables = query_api.query(query=query)
            
            records = []
            for table in tables:
                for row in table.records:
                    records.append({
                        "_time": row.get_time(),
                        "_field": row.get_field(),
                        "_value": row.get_value(),
                        "product_id": row.values.get("product_id"),
                        "operation_mode": row.values.get("operation_mode"),
                        "pump_health": row.values.get("pump_health"),
                        "pump_status": row.values.get("pump_status"),
                        "remote_operations": row.values.get("remote_operations"),
                        "rtc_health": row.values.get("rtc_health")
                    })
            
            return pd.DataFrame(records)
        finally:
            client.close()

    def fetch_running_flow(self, pump_id: str, days: int, window_minutes: int = 5) -> list[float]:
        """Flow readings (median per window) from the times the pump status was ON."""
        query = f'''
        from(bucket: {self._flux_string(self.bucket)})
          |> range(start: -{int(days)}d)
          |> filter(fn: (r) => r._measurement == {self._flux_string(self.measurement)})
          |> filter(fn: (r) => r.product_id == {self._flux_string(pump_id)})
          |> filter(fn: (r) => r._field == "flowrate")
          |> filter(fn: (r) => r.pump_status == "on" or r.pump_status == "ON" or r.pump_status == "running" or r.pump_status == "RUNNING")
          |> toFloat()
          |> group()
          |> sort(columns: ["_time"])
          |> aggregateWindow(every: {int(window_minutes)}m, fn: median, createEmpty: false)
        '''
        client = self._get_client()
        try:
            tables = client.query_api().query(query=query)
            return [float(record.get_value()) for table in tables for record in table.records if record.get_value() is not None]
        finally:
            client.close()

    def get_latest_status_row(self, pump_id: str, start: str = "0", *, bounded: bool = False) -> dict | None:
        """Fetch the latest telemetry tag row without requiring valid sensor fields."""
        if bounded and start == "0":
            for window in self.status_lookback_minutes:
                try:
                    row = self.get_latest_status_row(pump_id, f"-{window}m")
                except Exception:
                    logger.exception("Bounded status query failed for %s", pump_id)
                    self._record_lookback(pump_id, "status", None, True)
                    return self.get_latest_status_row(pump_id)
                if row is not None:
                    self._record_lookback(pump_id, "status", window, False)
                    return row
            self._record_lookback(pump_id, "status", None, False)
            return self.get_latest_status_row(pump_id)
        query = f'''
        from(bucket: {self._flux_string(self.bucket)})
          |> range(start: {start})
          |> filter(fn: (r) => r._measurement == {self._flux_string(self.measurement)})
          |> filter(fn: (r) => r.product_id == {self._flux_string(pump_id)})
          |> last()
          |> group()
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: 1)
        '''
        client = self._get_client()
        try:
            result = client.query_api().query(query=query)
            for table in result:
                for row in table.records:
                    return {
                        "_time": row.get_time(),
                        "product_id": row.values.get("product_id"),
                        "operation_mode": row.values.get("operation_mode"),
                        "pump_health": row.values.get("pump_health"),
                        "pump_status": row.values.get("pump_status"),
                    }
            return None
        finally:
            client.close()

    def get_wide_dataframe(self, pump_id: Optional[str] = None, start: str = "-7d", stop: str = "now()", tail_count: Optional[int] = None, on_only: bool = False) -> pd.DataFrame:
        """
        Fetches raw data from the configured measurement and transforms it into a wide format DataFrame
        ready for the ML and prediction pipeline.
        """
        raw_tail = None if on_only else tail_count
        raw_df = self.fetch_raw_data(pump_id, start, stop, raw_tail, on_only=on_only)
        
        if raw_df.empty:
            logger.warning(f"No data returned for pump {pump_id} in measurement '{self.measurement}'.")
            return pd.DataFrame()

        # Data Quality Firewall: Record exact fields that sent '-' (missing/corrupted non-numeric data)
        hyphen_mask = raw_df['_value'].astype(str).str.strip() == '-'
        if pump_id:
            self.hyphen_fields[pump_id] = raw_df.loc[hyphen_mask, '_field'].unique().tolist()
        raw_df['_is_hyphen'] = hyphen_mask

        # Preserve missing/corrupt sensor values as NaN so the model reports DATA_ERROR.
        raw_df['_value'] = pd.to_numeric(raw_df['_value'], errors='coerce')
        
        # Tags we want to preserve in the wide format
        index_cols = ["_time", "product_id", "operation_mode", "pump_health", "pump_status"]
        for col in index_cols:
            if col in raw_df.columns:
                raw_df[col] = raw_df[col].fillna('unknown')

        # Safely align staggered timestamps to the nearest second
        raw_df['_time'] = raw_df['_time'].dt.round('s')

        # Drop exact duplicates
        raw_df = raw_df.drop_duplicates(subset=["_time", "_field"])
        
        # Pivot Long to Wide
        wide_df = raw_df.pivot(index=index_cols, columns="_field", values="_value").reset_index()
        wide_df = wide_df.rename_axis(None, axis=1)
        hyphen_rows = raw_df.loc[raw_df['_is_hyphen']].groupby('_time')['_field'].agg(lambda names: ",".join(sorted(set(names))))
        wide_df[HYPHEN_COLUMN] = wide_df['_time'].map(hyphen_rows).fillna("")

        # Keep absent channels missing; zero is a real physical reading, not a fill value.
        for f in REQUIRED_SENSOR_FIELDS:
            if f not in wide_df.columns:
                wide_df[f] = float('nan')
        mark_nonfinite_rows(wide_df, REQUIRED_SENSOR_FIELDS)

        # Automatically normalize pump_status
        if 'pump_status' in wide_df.columns and 'current_b' in wide_df.columns:
            off_mask = (wide_df['current_b'] <= 0.5) & (wide_df['pump_status'].str.lower().isin(['unknown', 'off', 'offline', '']))
            wide_df.loc[off_mask, 'pump_status'] = 'OFF'

        wide_df = wide_df.sort_values("_time")
        if tail_count:
            wide_df = wide_df.tail(tail_count)

        return add_average_features(wide_df.reset_index(drop=True))

    def get_latest_live_on_dataframe(self, pump_id: str, limit: int = 350, *, bounded: bool = False, _start: str = "0") -> pd.DataFrame:
        """
        Fetches the most recent 'limit' number of ON readings directly from InfluxDB
        using a server-side pivot. This avoids fetching massive amounts of history over the network.
        """
        if bounded and limit > 0:
            probe_limit = limit + LOOKBACK_PROBE_MARGIN
            for window in self.live_lookback_minutes:
                try:
                    candidate = self.get_latest_live_on_dataframe(pump_id, probe_limit, _start=f"-{window}m")
                except Exception:
                    logger.exception("Bounded live query failed for %s", pump_id)
                    self._record_lookback(pump_id, "live_history", None, True)
                    return self.get_latest_live_on_dataframe(pump_id, limit)
                if len(candidate) >= probe_limit and candidate["_time"].is_monotonic_increasing:
                    if candidate["_time"].iloc[-limit] != candidate["_time"].iloc[-limit - 1]:
                        self._record_lookback(pump_id, "live_history", window, False)
                        return candidate.tail(limit).reset_index(drop=True)
            self._record_lookback(pump_id, "live_history", None, False)
            return self.get_latest_live_on_dataframe(pump_id, limit)
        required = " or ".join(
            f"r._field == {self._flux_string(field)}"
            for field in REQUIRED_SENSOR_FIELDS
        )
        query = f'''
        from(bucket: {self._flux_string(self.bucket)})
          |> range(start: {_start})
          |> filter(fn: (r) => r._measurement == {self._flux_string(self.measurement)})
          |> filter(fn: (r) => r.product_id == {self._flux_string(pump_id)})
          |> filter(fn: (r) => r.pump_status == "on" or r.pump_status == "ON" or r.pump_status == "running" or r.pump_status == "RUNNING")
          |> filter(fn: (r) => ({required}))
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: {limit})
          |> sort(columns: ["_time"], desc: false)
        '''
        logger.info(f"Querying InfluxDB for {pump_id} latest {limit} ON readings (live inference fetch).")
        client = self._get_client()
        try:
            tables = client.query_api().query_data_frame(query=query)
            if isinstance(tables, list):
                if not tables:
                    return pd.DataFrame()
                wide_df = pd.concat(tables, ignore_index=True)
            else:
                wide_df = tables

            if wide_df.empty:
                return wide_df

            # Data Quality Firewall: track hyphen fields per row before coercion.
            hyphen_cols = _mark_hyphen_rows(wide_df, REQUIRED_SENSOR_FIELDS)
            if hyphen_cols:
                self.hyphen_fields[pump_id] = hyphen_cols
            
            # Ensure required numeric fields are numeric
            for f in REQUIRED_SENSOR_FIELDS:
                if f in wide_df.columns:
                    wide_df[f] = pd.to_numeric(wide_df[f], errors='coerce')
                else:
                    wide_df[f] = float('nan')
            mark_nonfinite_rows(wide_df, REQUIRED_SENSOR_FIELDS)

            # Drop result/table cols from influx
            cols_to_drop = [c for c in ['result', 'table', '_start', '_stop'] if c in wide_df.columns]
            wide_df = wide_df.drop(columns=cols_to_drop)

            # Keep tags safe
            index_cols = ["_time", "product_id", "operation_mode", "pump_health", "pump_status"]
            for col in index_cols:
                if col in wide_df.columns:
                    wide_df[col] = wide_df[col].fillna('unknown')

            if '_time' in wide_df.columns:
                # Need to convert string/timestamp to proper timezone-aware datetime before rounding
                wide_df['_time'] = pd.to_datetime(wide_df['_time'], utc=True).dt.round('s')
                
            wide_df = wide_df.sort_values("_time").drop_duplicates("_time", keep="last")
            if limit:
                wide_df = wide_df.tail(limit)
            
            return add_average_features(wide_df.reset_index(drop=True))
        finally:
            client.close()

    def get_recent_wide_dataframe(self, pump_id: str, start: str = "-90m", limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch a small recent telemetry window without downloading full history.

        This is used by realtime engineering checks.  It intentionally does not
        filter the ``pump_status`` tag so an OFF reading or an incorrectly tagged
        OFF reading remains visible to the physical-state checks.
        """
        required = " or ".join(
            f"r._field == {self._flux_string(field)}"
            for field in REQUIRED_SENSOR_FIELDS
        )
        limit_clause = f'\n          |> limit(n: {int(limit)})' if limit else ""
        query = f'''
        from(bucket: {self._flux_string(self.bucket)})
          |> range(start: {start})
          |> filter(fn: (r) => r._measurement == {self._flux_string(self.measurement)})
          |> filter(fn: (r) => r.product_id == {self._flux_string(pump_id)})
          |> filter(fn: (r) => ({required}))
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> sort(columns: ["_time"], desc: true){limit_clause}
          |> sort(columns: ["_time"], desc: false)
        '''
        logger.info("Querying recent telemetry for %s (start=%s, limit=%s).", pump_id, start, limit)
        client = self._get_client()
        try:
            tables = client.query_api().query_data_frame(query=query)
            if isinstance(tables, list):
                if not tables:
                    return pd.DataFrame()
                wide_df = pd.concat(tables, ignore_index=True)
            else:
                wide_df = tables
            if wide_df.empty:
                return wide_df

            hyphen_fields = _mark_hyphen_rows(wide_df, REQUIRED_SENSOR_FIELDS)
            for field in REQUIRED_SENSOR_FIELDS:
                if field in wide_df.columns:
                    wide_df[field] = pd.to_numeric(wide_df[field], errors='coerce')
                else:
                    wide_df[field] = float('nan')
            mark_nonfinite_rows(wide_df, REQUIRED_SENSOR_FIELDS)
            if hyphen_fields:
                self.hyphen_fields[pump_id] = hyphen_fields

            cols_to_drop = [column for column in ['result', 'table', '_start', '_stop'] if column in wide_df.columns]
            wide_df = wide_df.drop(columns=cols_to_drop)
            for column in ["product_id", "operation_mode", "pump_health", "pump_status"]:
                if column in wide_df.columns:
                    wide_df[column] = wide_df[column].fillna('unknown')
                elif column == "pump_status":
                    wide_df[column] = 'unknown'

            wide_df['_time'] = pd.to_datetime(wide_df['_time'], utc=True, errors='coerce').dt.round('s')
            wide_df = wide_df.dropna(subset=['_time']).sort_values('_time').drop_duplicates('_time', keep='last')
            if limit:
                wide_df = wide_df.tail(limit)
            return add_average_features(wide_df.reset_index(drop=True))
        finally:
            client.close()

    def get_hourly_forecast_dataframe(self, pump_id: str, start: str = "-90d", profile: dict | None = None) -> pd.DataFrame:
        """Return hourly electrical means from usable status-ON rows only."""
        raw = self.get_wide_dataframe(pump_id, start=start)
        return self.prepare_hourly_forecast_dataframe(raw, profile)

    @staticmethod
    def prepare_hourly_forecast_dataframe(raw: pd.DataFrame, profile: dict | None = None) -> pd.DataFrame:
        """Build hourly forecast input from an already-fetched raw history frame."""
        if raw.empty:
            return raw
        fields = ("current_r", "current_y", "current_b", "voltage_r", "voltage_y", "voltage_b", "pf_r", "pf_y", "pf_b", "total_kw")
        hourly = aggregate_on_only(
            forecast_input_firewall(raw, profile),
            interval="1h",
            target_fields=fields,
            min_on_samples=settings.FORECAST_ON_MIN_SAMPLES_HOURLY,
        ).reset_index(names="_time")
        return add_average_features(hourly)

    def get_cached_hourly_forecast_dataframe(self, pump_id: str, profile_version: int, start: str = "-90d") -> pd.DataFrame:
        """Read version-matched precomputed hourly forecast rows from InfluxDB."""
        query = f'''
        from(bucket: {self._flux_string(settings.FORECAST_HOURLY_BUCKET)})
          |> range(start: {start})
          |> filter(fn: (r) => r._measurement == {self._flux_string(settings.FORECAST_HOURLY_MEASUREMENT)})
          |> filter(fn: (r) => r.product_id == {self._flux_string(pump_id)})
          |> filter(fn: (r) => r.profile_version == {self._flux_string(str(profile_version))})
          |> filter(fn: (r) => r.pipeline_version == {self._flux_string(str(settings.FORECAST_HOURLY_PIPELINE_VERSION))})
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> group()
          |> sort(columns: ["_time"])
        '''
        client = self._get_client()
        try:
            frame = client.query_api().query_data_frame(query=query)
            if isinstance(frame, list):
                frame = pd.concat(frame, ignore_index=True) if frame else pd.DataFrame()
            if frame.empty:
                return frame
            frame = frame.drop(columns=[column for column in ("result", "table", "_start", "_stop") if column in frame], errors="ignore")
            frame["_time"] = pd.to_datetime(frame["_time"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["_time"]).sort_values("_time").drop_duplicates("_time", keep="last")
            for field in (*HOURLY_FORECAST_FIELDS, *HOURLY_DIAGNOSTIC_FIELDS):
                if field not in frame:
                    frame[field] = float("nan")
                frame[field] = pd.to_numeric(frame[field], errors="coerce")
            return add_average_features(frame.reset_index(drop=True))
        finally:
            client.close()

    def write_hourly_forecast_dataframe(self, pump_id: str, profile_version: int, frame: pd.DataFrame) -> int:
        """Idempotently upsert hourly forecast inputs using pump and hour identity."""
        if frame.empty:
            return 0
        points = []
        for _, row in frame.iterrows():
            timestamp = pd.to_datetime(row.get("_time"), utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            point = (
                Point(settings.FORECAST_HOURLY_MEASUREMENT)
                .tag("product_id", pump_id)
                .tag("profile_version", str(profile_version))
                .tag("pipeline_version", str(settings.FORECAST_HOURLY_PIPELINE_VERSION))
                .time(timestamp.to_pydatetime())
            )
            field_count = 0
            for field in (*HOURLY_FORECAST_FIELDS, *HOURLY_DIAGNOSTIC_FIELDS):
                value = pd.to_numeric(row.get(field), errors="coerce")
                if pd.isna(value):
                    continue
                if field in {"on_sample_count", "status_on_samples", "usable_on_samples", "invalid_on_samples", "no_draw_samples"}:
                    point = point.field(field, int(value))
                else:
                    point = point.field(field, float(value))
                field_count += 1
            if field_count:
                points.append(point)
        if not points:
            return 0
        client = self._get_client()
        try:
            with client.write_api(write_options=SYNCHRONOUS) as writer:
                writer.write(bucket=settings.FORECAST_HOURLY_BUCKET, org=self.org, record=points)
        finally:
            client.close()
        return len(points)

    def refresh_hourly_forecast_cache(self, pump_id: str, profile: dict) -> pd.DataFrame:
        """Incrementally rebuild recent hours and return up to 90 days of cache.

        Only whole hours are rebuilt: the query window starts one hour early and raw
        rows before an hour boundary are dropped, so a refresh can never overwrite a
        cached hour with an aggregate of a partial hour.
        """
        profile_version = int(profile.get("version", 0))
        now = pd.Timestamp(datetime.now(timezone.utc))
        oldest_complete_hour = (now - timedelta(days=90)).ceil("h")
        cached = self.get_cached_hourly_forecast_dataframe(pump_id, profile_version, "-90d")
        latest = pd.to_datetime(cached["_time"].max(), utc=True, errors="coerce") if not cached.empty else pd.NaT
        if pd.isna(latest):
            boundary = oldest_complete_hour
        else:
            reprocess = timedelta(hours=max(0, int(settings.FORECAST_HOURLY_REPROCESS_HOURS)))
            boundary = max(latest.floor("h") - reprocess, oldest_complete_hour)
        lookback_hours = min(2160, max(1, math.ceil((now - boundary).total_seconds() / 3600) + 1))
        raw = self.get_wide_dataframe(pump_id, start=f"-{lookback_hours}h")
        if not raw.empty:
            raw = raw.loc[pd.to_datetime(raw["_time"], utc=True, errors="coerce") >= boundary].reset_index(drop=True)
        updated = self.prepare_hourly_forecast_dataframe(raw, profile)
        self.write_hourly_forecast_dataframe(pump_id, profile_version, updated)
        return self.get_cached_hourly_forecast_dataframe(pump_id, profile_version, "-90d")
