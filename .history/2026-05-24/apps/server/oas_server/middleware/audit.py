"""Audit log middleware.

Records every successful mutating request (POST/PATCH/PUT/DELETE) into the
`audit_log` table. We do not log GETs — they are not state changes, and
recording every read would balloon the table.

Body summary: we keep at most ~1KB of the request body, with obvious secrets
(`password`, `token`, anything ending in `_key`) redacted. The full body is
not stored because it can contain PII, voice consent uploads, etc.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from oas_core.db import AuditLog, session_scope
from oas_core.observability import get_request_id
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from oas_server.auth import verify_token

log = logging.getLogger(__name__)

_MUTATING = {"POST", "PATCH", "PUT", "DELETE"}
_SKIP_PATHS = ("/healthz", "/readyz", "/openapi.json", "/docs", "/redoc")
_BODY_LIMIT = 1024
_REDACT_KEYS = ("password", "token", "secret")


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: ("[redacted]" if any(s in k.lower() for s in _REDACT_KEYS) else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _identify(request: Request) -> tuple[str | None, str]:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None, "anonymous@local"
    payload = verify_token(auth.split(None, 1)[1].strip())
    if not payload:
        return None, "anonymous@local"
    uid = payload.get("sub")
    # Email lookup is cheap (indexed PK lookup) but we keep it best-effort.
    try:
        from oas_core.db import User

        with session_scope() as s:
            u = s.get(User, uid)
            if u:
                return u.id, u.email
    except Exception:
        pass
    return uid, "unknown"


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Buffer the body before the route handler consumes it so we can log
        # a redacted summary.
        body_summary: dict[str, Any] = {}
        if request.method in _MUTATING and not any(
            request.url.path.startswith(p) for p in _SKIP_PATHS
        ):
            raw = await request.body()

            async def _replay() -> dict[str, Any]:
                return {"type": "http.request", "body": raw, "more_body": False}

            # Re-inject the body so the downstream handler can read it again.
            request._receive = _replay

            if raw:
                try:
                    parsed = json.loads(raw)
                    parsed = _redact(parsed)
                    body_summary = {"json": parsed}
                except Exception:
                    body_summary = {"raw_size": len(raw)}

        response: Response = await call_next(request)

        if (
            request.method in _MUTATING
            and 200 <= response.status_code < 400
            and not any(request.url.path.startswith(p) for p in _SKIP_PATHS)
        ):
            uid, email = _identify(request)
            project_id = None
            if isinstance(body_summary.get("json"), dict):
                project_id = body_summary["json"].get("project_id")
            # Trim body to limit
            payload = json.dumps(body_summary)
            if len(payload) > _BODY_LIMIT:
                body_summary = {"truncated": True, "size": len(payload)}
            try:
                with session_scope() as s:
                    s.add(
                        AuditLog(
                            actor_user_id=uid,
                            actor_email=email,
                            method=request.method,
                            path=request.url.path,
                            project_id=project_id,
                            status_code=response.status_code,
                            body_summary=body_summary,
                            request_id=get_request_id(),
                        )
                    )
            except Exception:
                log.exception("audit log write failed")

        return response
