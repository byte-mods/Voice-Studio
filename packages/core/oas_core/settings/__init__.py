"""Typed settings loaded from environment variables.

Read once at process start; pass the resulting `Settings` instance to anything
that needs config. Do not call `get_settings()` from hot paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    db_url: str
    log_level: str
    server_host: str
    server_port: int
    cors_origins: tuple[str, ...]
    hf_token: str | None
    wandb_api_key: str | None
    worker_concurrency: int
    enable_gpu: bool
    job_backend: str
    ray_address: str | None

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_env("OAS_DATA_DIR", "./data")).resolve()
    db_url = _env("OAS_DB_URL", f"sqlite:///{data_dir / 'oas.db'}")
    cors = tuple(
        s.strip() for s in _env("OAS_CORS_ORIGINS", "http://localhost:3000").split(",") if s.strip()
    )
    return Settings(
        data_dir=data_dir,
        db_url=db_url,
        log_level=_env("OAS_LOG_LEVEL", "INFO"),
        server_host=_env("OAS_SERVER_HOST", "0.0.0.0"),
        server_port=_env_int("OAS_SERVER_PORT", 8000),
        cors_origins=cors,
        hf_token=os.environ.get("HF_TOKEN") or None,
        wandb_api_key=os.environ.get("WANDB_API_KEY") or None,
        worker_concurrency=_env_int("OAS_WORKER_CONCURRENCY", 2),
        enable_gpu=_env_bool("OAS_ENABLE_GPU", True),
        job_backend=_env("OAS_JOB_BACKEND", "inprocess"),
        ray_address=os.environ.get("OAS_RAY_ADDRESS") or None,
    )


def ensure_dirs(s: Settings) -> None:
    for d in (s.data_dir, s.artifacts_dir, s.datasets_dir, s.models_dir, s.runs_dir):
        d.mkdir(parents=True, exist_ok=True)
