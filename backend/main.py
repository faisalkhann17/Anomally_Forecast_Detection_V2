"""FastAPI entrypoint: register routes and start pump discovery/background jobs."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

from backend import __version__
from backend.config import settings
from backend.api.routes import admin, pumps, profiles
from backend.workers.prediction_worker import worker_instance

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

app = FastAPI(
    title=settings.API_TITLE,
    version=__version__,
    description="Backend for Pump Predictive Maintenance Dashboard"
)

# Permit the configured dashboard origins and only the methods/headers used by the API.
origins = [origin.strip() for origin in settings.ALLOWED_CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Keep operator pump reads, admin diagnostics and engineer profiles on separate routes.
app.include_router(pumps.router)
app.include_router(admin.router)
app.include_router(profiles.router)

@app.on_event("startup")
async def startup_event():
    """Load stored profiles, discover pump IDs, then start forecast/maintenance loops."""
    if settings.AUTH_DISABLED_FOR_DEV:
        logging.getLogger(__name__).warning("AUTH_DISABLED_FOR_DEV is enabled; protected routes are open.")
    if not settings.ADMIN_API_KEY or not (settings.ENGINEER_API_KEY or settings.ENGINEER_API_KEYS):
        logging.getLogger(__name__).warning("One or more API keys are missing; affected protected routes fail closed.")
    try:
        # Warm the in-memory copy so live checks survive a later MongoDB outage.
        await asyncio.to_thread(worker_instance.profile_repository.load_all)
        recorded = await asyncio.to_thread(worker_instance.profile_repository.backfill_history)
        if recorded:
            logging.getLogger(__name__).info("Recorded %d profile version(s) missing from profile history.", recorded)
    except Exception:
        logging.getLogger(__name__).warning("MongoDB profile store is unavailable at startup; profile saves will fail until it is reachable.")
    # The pump list is fetched from Influx only when GET /api/pumps is called;
    # scheduled jobs run for pumps with approved engineering profiles.
    await worker_instance.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cancel scheduled work when the API process shuts down."""
    worker_instance.stop()

@app.get("/")
async def root():
    """Return basic API identity and version information."""
    return {
        "service": "PumpWatch Predictive Maintenance API",
        "version": __version__,
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Report model, event-store and worker readiness for monitoring."""
    model_ready = worker_instance.hourly_timestral.load_error is None
    issue_event_store_ready = worker_instance.issue_event_repository is not None and worker_instance.issue_event_store_error is None
    profile_store_ready = getattr(worker_instance.profile_repository, "last_error", None) is None
    return {
        "status": "ok" if model_ready and profile_store_ready else "degraded",
        "message": "PumpWatch Backend is running",
        "baseline_store": "retired",
        "model_ready": model_ready,
        "profile_store": "mongodb",
        "profile_store_ready": profile_store_ready,
        "profile_history_ready": getattr(worker_instance.profile_repository, "history_error", None) is None,
        "issue_event_store_ready": issue_event_store_ready,
        "forecast_model": "Timestral-123M",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
