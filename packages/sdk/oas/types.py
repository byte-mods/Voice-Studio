"""Typed response models for the SDK.

Kept intentionally permissive (extra='allow') so SDK consumers don't break when
the server adds new fields between releases.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")


class Project(_Base):
    id: str
    slug: str
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime
    settings: dict[str, Any] = {}


class Dataset(_Base):
    id: str
    project_id: str
    slug: str
    name: str
    modality: str
    description: str | None = None
    source: str | None = None
    tags: list[str] = []
    created_at: datetime
    updated_at: datetime


class DatasetVersion(_Base):
    id: str
    dataset_id: str
    version: str
    manifest_uri: str
    num_samples: int
    total_audio_s: float
    stats: dict[str, Any] = {}
    notes: str | None = None
    created_at: datetime


class Run(_Base):
    id: str
    job_id: str
    attempt: int
    status: str
    metrics: dict[str, Any] = {}
    logs_uri: str | None = None
    artifacts_uri: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class Job(_Base):
    id: str
    project_id: str
    kind: str
    name: str
    config: dict[str, Any] = {}
    status: str
    priority: int = 0
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    runs: list[Run] = []


class Experiment(_Base):
    id: str
    project_id: str
    name: str
    description: str | None = None
    run_ids: list[str] = []
    created_at: datetime


class Model(_Base):
    id: str
    project_id: str
    slug: str
    name: str
    modality: str
    family: str | None = None
    description: str | None = None
    created_at: datetime


class ModelVersion(_Base):
    id: str
    model_id: str
    version: str
    stage: str
    artifact_uri: str
    format: str
    size_bytes: int
    sha256: str | None = None
    config: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    source_run_id: str | None = None
    source_dataset_version_id: str | None = None
    notes: str | None = None
    created_at: datetime
