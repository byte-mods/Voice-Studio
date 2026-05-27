"""Demonstration job handler.

Install this plugin (`pip install -e plugins/example`) and the studio will
auto-discover its `echo_plus` kind on next startup. No core changes required.
"""

from __future__ import annotations

from typing import Any

from oas_core.queue.backend import JobContext


def echo_plus(ctx: JobContext) -> dict[str, Any]:
    msg = ctx.config.get("msg", "hello from a plugin")
    times = int(ctx.config.get("times", 3))
    for i in range(times):
        ctx.log(f"[{i + 1}/{times}] {msg}")
        ctx.heartbeat()
    return {"echoed": msg, "times": times}
