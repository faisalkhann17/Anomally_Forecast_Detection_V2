"""Small API-key authorization and per-process request limiting."""
from __future__ import annotations

from collections import defaultdict, deque
import hmac
import time

from fastapi import Header, HTTPException, Request

from backend.config import settings


_requests: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_failed_requests: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Return the direct client address used by in-process rate limits."""
    return request.client.host if request.client else "unknown"


def _recent_attempts(bucket: deque[float]) -> float:
    """Drop attempts older than one minute and return monotonic time."""
    now = time.monotonic()
    while bucket and bucket[0] <= now - 60:
        bucket.popleft()
    return now


def _check_failed_rate_limit(request: Request) -> None:
    """Block clients exceeding the failed-authentication attempt limit."""
    bucket = _failed_requests[_client_ip(request)]
    _recent_attempts(bucket)
    if len(bucket) >= max(1, settings.FAILED_AUTH_RATE_LIMIT_PER_MINUTE):
        raise HTTPException(status_code=429, detail="Authentication rate limit exceeded.")


def _record_failed_auth(request: Request) -> None:
    """Record one rejected authentication attempt for the client."""
    _failed_requests[_client_ip(request)].append(time.monotonic())


def _check_rate_limit(identity: str, request: Request) -> None:
    """Apply the protected-route limit per identity and direct client IP."""
    client_ip = _client_ip(request)
    bucket = _requests[(identity, client_ip)]
    now = _recent_attempts(bucket)
    if len(bucket) >= max(1, settings.PROTECTED_RATE_LIMIT_PER_MINUTE):
        raise HTTPException(status_code=429, detail="Protected-route rate limit exceeded.")
    bucket.append(now)


def authorize(request: Request, api_key: str | None, role: str) -> str:
    """Authenticate a key, enforce role access and return its identity label."""
    if settings.AUTH_DISABLED_FOR_DEV:
        identity = "dev"
    elif not settings.API_KEY_REQUIRED:
        identity = "admin"
    else:
        admin_key = settings.ADMIN_API_KEY
        engineer_key = settings.ENGINEER_API_KEY
        named_keys = []
        for entry in settings.ENGINEER_API_KEYS.split(","):
            name, separator, value = entry.partition(":")
            if separator and name.strip() and value.strip():
                named_keys.append((name.strip(), value.strip()))
        configured = bool(admin_key) if role == "admin" else bool(admin_key or engineer_key or named_keys)
        if not configured:
            raise HTTPException(status_code=503, detail="Auth not configured.")
        _check_failed_rate_limit(request)
        if not api_key:
            _record_failed_auth(request)
            raise HTTPException(status_code=401, detail="X-API-Key is required.")
        admin_match = bool(admin_key) and hmac.compare_digest(api_key, admin_key)
        engineer_match = bool(engineer_key) and hmac.compare_digest(api_key, engineer_key)
        named_matches = [(name, hmac.compare_digest(api_key, value)) for name, value in named_keys]
        if not admin_match and not (role == "engineer" and (engineer_match or any(match for _, match in named_matches))):
            _record_failed_auth(request)
            raise HTTPException(status_code=403, detail="Invalid API key.")
        identity = "admin" if admin_match else "engineer" if engineer_match else next(
            name for name, match in named_matches if match
        )
    _check_rate_limit(identity, request)
    return identity


def require_engineer(request: Request, x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """FastAPI dependency allowing engineer or administrator credentials."""
    return authorize(request, x_api_key, "engineer")


def require_admin(request: Request, x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """FastAPI dependency allowing administrator credentials only."""
    return authorize(request, x_api_key, "admin")
