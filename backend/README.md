# PumpWatch backend package

This folder is the Python package. Run Python commands from the repository
root so `backend.*` imports resolve.

## Layout

```text
backend/
├── api/         # HTTP routes, response shaping, authentication
├── config/      # Environment-based settings
├── engine/      # Electrical rules, health, data quality, trends, flow suggestion
├── services/    # InfluxDB, MongoDB and Timestral adapters
├── workers/     # Realtime refresh and scheduled forecast/maintenance jobs
├── artifacts/   # Runtime data and the model checkpoint (not in Git)
└── main.py      # FastAPI entrypoint
```

`api/DASHBOARD_CONTRACT.md` describes the dashboard response the frontend uses.
`.env` and the model checkpoint are environment-specific and are not committed.
