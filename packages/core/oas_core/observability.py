"""Request-scoped context + structured logging.

A `request_id` is generated per incoming request (or accepted from the client's
`X-Request-ID` header). All log records emitted inside that request carry the
id automatically, which makes server logs joinable with the audit table.

Workers also stamp logs with `job_id` and `run_id` via `bind(...)`.
"""

from __future__ import annotations

import logging
import secrets
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("oas_request_id", default=None)
_extra: ContextVar[dict[str, Any] | None] = ContextVar("oas_extra", default=None)


def new_request_id() -> str:
    return secrets.token_urlsafe(9)


def set_request_id(rid: str | None) -> None:
    _request_id.set(rid)


def get_request_id() -> str | None:
    return _request_id.get()


def bind(**fields: Any) -> None:
    """Merge fields into the current logging context."""
    current = dict(_extra.get() or {})
    current.update(fields)
    _extra.set(current)


def get_context() -> dict[str, Any]:
    out: dict[str, Any] = dict(_extra.get() or {})
    rid = _request_id.get()
    if rid:
        out["request_id"] = rid
    return out


class ContextFilter(logging.Filter):
    """Logging filter that injects the request context into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = get_context()
        for k, v in ctx.items():
            setattr(record, k, v)
        # Make the formatter easy: provide a single compact suffix.
        if ctx:
            record.ctx_suffix = " " + " ".join(f"{k}={v}" for k, v in ctx.items())
        else:
            record.ctx_suffix = ""
        return True


def configure_logging(level: str = "INFO") -> None:
    """Idempotent root-logger setup with the context filter installed."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    # Replace handlers so re-configuring (uvicorn --reload) doesn't double them.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.addFilter(ContextFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s:%(ctx_suffix)s %(message)s")
    )
    root.addHandler(handler)
