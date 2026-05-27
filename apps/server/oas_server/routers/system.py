"""System info: version, queued handlers, GPU detection."""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any

from fastapi import APIRouter
from oas_core.queue.backend import list_handlers

from oas_server import __version__

router = APIRouter(prefix="/system", tags=["system"])


def _detect_gpus() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=2.0,
        )
    except Exception:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "memory_total": parts[1], "driver": parts[2]})
    return gpus


@router.get("/info")
def info() -> dict[str, Any]:
    return {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "handlers": list_handlers(),
        "gpus": _detect_gpus(),
    }
