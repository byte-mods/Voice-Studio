"""Request ID middleware.

Honors an inbound `X-Request-ID` if provided (handy when the studio sits
behind a proxy that injects its own), otherwise generates one. The id is
stashed in a contextvar so log records and audit entries pick it up, and
echoed back on the response so clients can correlate.
"""

from __future__ import annotations

from oas_core.observability import new_request_id, set_request_id
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    HEADER = "x-request-id"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get(self.HEADER) or new_request_id()
        token = set_request_id(rid)  # type: ignore[func-returns-value]
        try:
            response = await call_next(request)
            response.headers[self.HEADER] = rid
            return response
        finally:
            # Reset to None so subsequent unrelated logs don't carry it.
            set_request_id(None)
            _ = token
