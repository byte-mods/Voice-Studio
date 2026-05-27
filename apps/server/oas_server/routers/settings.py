"""Settings endpoints: read sanitized settings, list available integrations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from oas_core.settings import get_settings

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("")
def read_settings() -> dict[str, Any]:
    s = get_settings()
    return {
        "data_dir": str(s.data_dir),
        "db_url": _redact(s.db_url),
        "log_level": s.log_level,
        "worker_concurrency": s.worker_concurrency,
        "enable_gpu": s.enable_gpu,
        "integrations": {
            "huggingface": bool(s.hf_token),
            "wandb": bool(s.wandb_api_key),
        },
    }


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    head, tail = url.rsplit("@", 1)
    if "://" in head:
        scheme, _ = head.split("://", 1)
        return f"{scheme}://***@{tail}"
    return f"***@{tail}"
