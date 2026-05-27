"""Built-in job handlers.

The studio ships with a few primitive handlers so the queue is functional out
of the box. Heavy ML handlers (HF import, ASR/LLM/TTS fine-tune) live in
`oas_server.jobs.*` and register themselves on import.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from oas_core.queue.backend import JobContext, register_handler


def _noop(ctx: JobContext) -> dict[str, Any]:
    ctx.log("noop handler executed")
    return {"ok": True}


def _sleep(ctx: JobContext) -> dict[str, Any]:
    seconds = float(ctx.config.get("seconds", 1.0))
    ctx.log(f"sleeping {seconds}s")
    for i in range(int(seconds * 10)):
        if ctx.cancelled:
            ctx.log("cancelled")
            break
        time.sleep(0.1)
        if i % 10 == 0:
            ctx.heartbeat()
    return {"slept_s": seconds}


for _kind, _handler in (("noop", _noop), ("sleep", _sleep)):
    with contextlib.suppress(ValueError):
        register_handler(_kind, _handler)

# Discover and register all built-in job-kind modules.
from oas_server import jobs  # noqa: E402,F401
