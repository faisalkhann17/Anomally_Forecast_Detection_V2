"""Environment-backed settings for Influx, model jobs, profile storage and API access."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")

class Settings(BaseSettings):
    """Load runtime settings from environment variables with safe defaults."""
    # API Settings
    API_TITLE: str = "PumpWatch Predictive Maintenance API"
    
    # InfluxDB Settings
    INFLUXDB_URL: str = os.getenv("INFLUXDB_URL", "")
    INFLUXDB_TOKEN: str = os.getenv("INFLUXDB_TOKEN", "")
    INFLUXDB_ORG: str = os.getenv("INFLUXDB_ORG", "Aquasense")
    INFLUXDB_BUCKET: str = os.getenv("INFLUXDB_BUCKET", "Flowmeter_Data_Influx")
    INFLUXDB_MEASUREMENT: str = os.getenv("INFLUXDB_MEASUREMENT", "atbc_packet")

    SHORT_CYCLING_LOOKBACK_MINUTES: int = int(os.getenv("SHORT_CYCLING_LOOKBACK_MINUTES", "60"))
    SHORT_CYCLING_MAX_STARTS: int = int(os.getenv("SHORT_CYCLING_MAX_STARTS", "4"))
    SHORT_CYCLING_MIN_ON_SECONDS: int = int(os.getenv("SHORT_CYCLING_MIN_ON_SECONDS", "30"))
    SHORT_CYCLING_MIN_OFF_SECONDS: int = int(os.getenv("SHORT_CYCLING_MIN_OFF_SECONDS", "30"))
    SHORT_CYCLING_MAX_GAP_SECONDS: int = int(os.getenv("SHORT_CYCLING_MAX_GAP_SECONDS", "300"))
    SHORT_CYCLING_MIN_OBSERVATION_MINUTES: int = int(os.getenv("SHORT_CYCLING_MIN_OBSERVATION_MINUTES", "45"))
    MAX_TELEMETRY_AGE_SECONDS: int = int(os.getenv("MAX_TELEMETRY_AGE_SECONDS", "900"))
    REFRESH_MIN_INTERVAL_SECONDS: int = int(os.getenv("REFRESH_MIN_INTERVAL_SECONDS", "10"))
    ADMIN_API_KEY: str = ""
    ENGINEER_API_KEY: str = ""
    ENGINEER_API_KEYS: str = ""
    AUTH_DISABLED_FOR_DEV: bool = False
    PROTECTED_RATE_LIMIT_PER_MINUTE: int = 60
    FAILED_AUTH_RATE_LIMIT_PER_MINUTE: int = 20
    # One background job refreshes the forecast and the slow-wear checks together.
    FORECAST_REFRESH_MINUTES: int = int(os.getenv("FORECAST_REFRESH_MINUTES", "720"))
    # CPU threads for the Timestral model. 1 keeps a core free for live dashboard
    # requests on small servers; 0 lets PyTorch use all cores.
    FORECAST_TORCH_THREADS: int = int(os.getenv("FORECAST_TORCH_THREADS", "1"))
    FORECAST_HOURLY_BUCKET: str = os.getenv("FORECAST_HOURLY_BUCKET", os.getenv("INFLUXDB_BUCKET", "Flowmeter_Data_Influx"))
    FORECAST_HOURLY_MEASUREMENT: str = os.getenv("FORECAST_HOURLY_MEASUREMENT", "pumpwatch_hourly_forecast")
    FORECAST_HOURLY_PIPELINE_VERSION: int = int(os.getenv("FORECAST_HOURLY_PIPELINE_VERSION", "2"))
    FORECAST_HOURLY_REPROCESS_HOURS: int = int(os.getenv("FORECAST_HOURLY_REPROCESS_HOURS", "3"))
    # INFLUX_WRITES_ENABLED=false: never write to InfluxDB (read-only). The hourly forecast
    # cache is then kept in memory and cache rows already in the bucket are ignored.
    INFLUX_WRITES_ENABLED: bool = os.getenv("INFLUX_WRITES_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
    # Days of raw history the background forecast cache reads (90 normally). Lower it to
    # limit query load on a large production InfluxDB.
    FORECAST_HISTORY_DAYS: int = max(3, min(90, int(os.getenv("FORECAST_HISTORY_DAYS", "90"))))
    FORECAST_ON_MIN_SAMPLES_5MIN: int = int(os.getenv("FORECAST_ON_MIN_SAMPLES_5MIN", "1"))
    FORECAST_ON_MIN_SAMPLES_HOURLY: int = int(os.getenv("FORECAST_ON_MIN_SAMPLES_HOURLY", "1"))
    MAINTENANCE_REFRESH_MINUTES: int = int(os.getenv("MAINTENANCE_REFRESH_MINUTES", "720"))
    # History for the forecast and slow-wear jobs is read one day per query, as one reading
    # per HISTORY_SAMPLE_MINUTES chosen by InfluxDB, with a short pause between queries.
    HISTORY_SAMPLE_MINUTES: int = max(1, int(os.getenv("HISTORY_SAMPLE_MINUTES", "5")))
    HISTORY_QUERY_PAUSE_SECONDS: float = float(os.getenv("HISTORY_QUERY_PAUSE_SECONDS", "0.2"))
    MAINTENANCE_HISTORY_DAYS: int = int(os.getenv("MAINTENANCE_HISTORY_DAYS", "90"))
    # "Check minimum flow from data" button: margin below the lowest normal running flow,
    # running hours needed, outlier share that triggers a warning, and the windows tried
    # in order (most recent first).
    FLOW_SUGGESTION_MARGIN_PERCENT: float = float(os.getenv("FLOW_SUGGESTION_MARGIN_PERCENT", "30"))
    FLOW_SUGGESTION_MIN_RUNNING_HOURS: float = float(os.getenv("FLOW_SUGGESTION_MIN_RUNNING_HOURS", "12"))
    FLOW_SUGGESTION_OUTLIER_WARNING_PERCENT: float = float(os.getenv("FLOW_SUGGESTION_OUTLIER_WARNING_PERCENT", "20"))
    # "Calculate limits from data" button: days read (one small query per day), the
    # reading interval InfluxDB returns, and running hours needed.
    LIMIT_SUGGESTION_DAYS: int = max(3, min(60, int(os.getenv("LIMIT_SUGGESTION_DAYS", "30"))))
    LIMIT_SUGGESTION_SAMPLE_MINUTES: int = max(5, int(os.getenv("LIMIT_SUGGESTION_SAMPLE_MINUTES", "15")))
    LIMIT_SUGGESTION_MIN_RUNNING_HOURS: float = float(os.getenv("LIMIT_SUGGESTION_MIN_RUNNING_HOURS", "24"))
    FLOW_SUGGESTION_LOOKBACK_DAYS: tuple[int, ...] = tuple(int(item) for item in os.getenv("FLOW_SUGGESTION_LOOKBACK_DAYS", "7,30").split(",") if item.strip())
    MAINTENANCE_BASELINE_HOURLY_POINTS: int = int(os.getenv("MAINTENANCE_BASELINE_HOURLY_POINTS", "168"))
    MAINTENANCE_RECENT_HOURLY_POINTS: int = int(os.getenv("MAINTENANCE_RECENT_HOURLY_POINTS", "72"))
    MAINTENANCE_PERSISTENCE_HOURLY_POINTS: int = int(os.getenv("MAINTENANCE_PERSISTENCE_HOURLY_POINTS", "12"))
    # Early warning when a run is this far (%) from the pump's usual level, the
    # median of its ON hours over the last EARLY_WARNING_LONG_TERM_DAYS days.
    EARLY_WARNING_LONG_TERM_PERCENT: float = float(os.getenv("EARLY_WARNING_LONG_TERM_PERCENT", "15"))
    EARLY_WARNING_LONG_TERM_DAYS: int = int(os.getenv("EARLY_WARNING_LONG_TERM_DAYS", "7"))
    EARLY_WARNING_LONG_TERM_MAX_AGE_HOURS: int = int(os.getenv("EARLY_WARNING_LONG_TERM_MAX_AGE_HOURS", "24"))

    # Per-pump baseline persistence. Secrets must be supplied through the
    # environment and are never sent to the browser.
    BASELINE_STORE: str = os.getenv("BASELINE_STORE", "json")
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "Ftu_localDB")
    MONGODB_BASELINE_COLLECTION: str = os.getenv(
        "MONGODB_BASELINE_COLLECTION", "pumpwatch_baselines"
    )
    MONGODB_EVENT_COLLECTION: str = os.getenv("MONGODB_EVENT_COLLECTION", "pumpwatch_issue_events")
    MONGODB_TIMEOUT_MS: int = int(os.getenv("MONGODB_TIMEOUT_MS", "5000"))
    # Engineering profiles (limits) are stored only in this MongoDB collection.
    PROFILE_MONGODB_URI: str = "mongodb://127.0.0.1:27017"
    PROFILE_MONGODB_DATABASE: str = "Ftu_localDB"
    MONGODB_PROFILE_COLLECTION: str = "pumpwatch_profiles"
    MONGODB_PROFILE_HISTORY_COLLECTION: str = "pumpwatch_profile_history"
    PROFILE_MONGODB_TIMEOUT_MS: int = 2000
    # Engineer profiles and issue events must use a local MongoDB unless this is explicitly set to true.
    PROFILE_MONGODB_ALLOW_REMOTE: bool = False
    # Profile sanity warnings, relative to each pump's own nameplate. Limits outside
    # these bounds are not rejected; the engineer must confirm them before saving.
    PROFILE_VOLTAGE_TOLERANCE_PERCENT: float = float(os.getenv("PROFILE_VOLTAGE_TOLERANCE_PERCENT", "25"))
    PROFILE_MAX_TO_RATED_RATIO: float = float(os.getenv("PROFILE_MAX_TO_RATED_RATIO", "1.5"))
    PROFILE_MIN_MAX_TO_RATED_RATIO: float = float(os.getenv("PROFILE_MIN_MAX_TO_RATED_RATIO", "0.5"))
    
    ALLOWED_CORS_ORIGINS: str = os.getenv(
        "ALLOWED_CORS_ORIGINS", "http://localhost:5173"
    )

    class Config:
        """Keep environment variable names case-sensitive."""
        case_sensitive = True

settings = Settings()
