# PumpWatch backend

FastAPI service that monitors pumps from InfluxDB telemetry: live engineering
checks against engineer-approved limits, early warnings, slow-wear trends, a
Timestral forecast, and issue history in MongoDB. The dashboard frontend is a
separate project and calls this API under `/api`.

## Setup

1. Tested with Python 3.13. Install dependencies:

   ```bash
   pip install -r backend/requirements.txt
   pip install -r backend/requirements-timestral.txt   # forecasting
   ```

2. Copy `backend/.env.example` to `backend/.env` and fill in the real InfluxDB,
   MongoDB and API-key values. Never commit `.env`.
   `ENGINEER_API_KEY` and `ADMIN_API_KEY` are required for the limits APIs.

3. Download the forecast model (about 490 MB, not stored in Git):

   ```bash
   huggingface-cli download mistralai/Timestral-123M --local-dir backend/artifacts/timestral/checkpoints/Timestral-123M
   ```

   Without the model, live checks, issue history and slow-wear checks still run;
   only the forecast reports that the model is unavailable.

## Run

From the repository root:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Engineer limits

Every profile field is required, including the minimum delivery flow (must be
above 0). `GET /api/pumps/{id}/profile/flow-suggestion` suggests that minimum
from the pump's running flow: 7 days (30 if under 12 running hours), readings
below 10% of the usual flow and IQR outliers removed, lowest remaining flow
minus 30%. Tune with `FLOW_SUGGESTION_MARGIN_PERCENT` (30),
`FLOW_SUGGESTION_MIN_RUNNING_HOURS` (12),
`FLOW_SUGGESTION_OUTLIER_WARNING_PERCENT` (20) and
`FLOW_SUGGESTION_LOOKBACK_DAYS` (`7,30`).

## Calculate limits from data

`GET /api/pumps/{id}/profile/limit-suggestion` reads the pump's last 30 days of running
data (one small query per day) and suggests rated power, rated current and operating
limits, with warnings (for example supply voltage outside ±10% or a negative power
factor) and the alarms those limits would have raised. It saves nothing.

## Production settings

```env
INFLUX_WRITES_ENABLED=false          # read-only on InfluxDB
PROFILE_MONGODB_ALLOW_REMOTE=true    # profiles and issue history on the MongoDB server
FORECAST_HISTORY_DAYS=30
MAINTENANCE_HISTORY_DAYS=30
FORECAST_REFRESH_MINUTES=720         # forecast + slow-wear every 12 hours
```

Run a single worker (`--workers 1`): the background job runs inside the process.

## Versions and releases

The running version is in `backend/__init__.py` (`__version__`) and is shown at
`GET /` and in `/docs`. Changes are listed in `CHANGELOG.md`, and each release
commit is tagged `vMAJOR.MINOR.PATCH`.

To release:

1. Update `__version__` in `backend/__init__.py`.
2. Add the changes to `CHANGELOG.md` under the new version and date.
3. Commit, then tag and push: `git tag -a v1.1.0 -m "v1.1.0"` and `git push origin main --tags`.

On the server, deploy a specific version with `git fetch --tags` and
`git checkout v1.1.0`, then restart. To roll back, check out the previous tag
and restart. `.env` and the model folder are not tracked, so they are kept.

See `backend/README.md` for the code layout.
