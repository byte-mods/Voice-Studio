"""Per-identity rate limiter for inference endpoints.

Token bucket algorithm, in-process. Buckets are keyed by:
  - the JWT subject (`sub`) when a valid bearer token is present,
  - otherwise the client IP.

Defaults are intentionally generous; tune via env:
  OAS_SERVE_RATE_RPS=2       # refill rate
  OAS_SERVE_RATE_BURST=10    # bucket size
  OAS_SERVE_RATE_SCOPE=/serve  # path prefix this applies to
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from oas_server.auth import verify_token


class _Bucket:
    __slots__ = ("last", "tokens")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last = time.monotonic()


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.rps = float(os.environ.get("OAS_SERVE_RATE_RPS", "2"))
        self.burst = float(os.environ.get("OAS_SERVE_RATE_BURST", "10"))
        self.scope = os.environ.get("OAS_SERVE_RATE_SCOPE", "/serve")
        self._buckets: dict[str, _Bucket] = defaultdict(lambda: _Bucket(self.burst))
        self._lock = Lock()

    def _identity(self, request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            payload = verify_token(auth.split(None, 1)[1].strip())
            if payload and payload.get("sub"):
                return f"u:{payload['sub']}"
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not request.url.path.startswith(self.scope):
            return await call_next(request)
        key = self._identity(request)
        now = time.monotonic()
        with self._lock:
            b = self._buckets[key]
            elapsed = now - b.last
            b.tokens = min(self.burst, b.tokens + elapsed * self.rps)
            b.last = now
            if b.tokens < 1:
                retry_after = max(1, int((1 - b.tokens) / max(self.rps, 0.01)))
                return JSONResponse(
                    {"detail": "rate limit exceeded", "retry_after_s": retry_after},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            b.tokens -= 1
        return await call_next(request)
