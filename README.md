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

See `backend/README.md` for the code layout.
