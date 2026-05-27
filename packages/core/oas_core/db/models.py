"""SQLAlchemy ORM models.

Conventions:
- IDs are short ULIDs (k-sortable, URL-safe) stored as strings.
- Timestamps are UTC `datetime`.
- All blobs/configs are stored as JSON columns (SQLite JSON1 + Postgres JSONB).
- Cross-entity references use foreign keys; cascade deletes are off by default
  so destroying a project is an explicit, audited action.
"""

from __future__ import annotations

import enum
import secrets
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Re-export the canonical Modality enum defined in the manifest schema so the
# DB layer and the manifest layer agree on a single type. A parallel local
# enum previously caused arg-type collisions at the DB <-> manifest seam
# (e.g. ManifestHeader.modality at routers/datasets.py:295).
from oas_core.manifest.schema import (
    Modality as Modality,  # explicit re-export for mypy --no-implicit-reexport
)


def _new_id() -> str:
    # 16 bytes of urlsafe randomness; collision-free at our scale.
    return secrets.token_urlsafe(12)


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    # SQLAlchemy reads this class-level mapping during model introspection; it is
    # never mutated. RUF012's mutable-default warning does not apply.
    type_annotation_map = {dict[str, Any]: JSON}  # noqa: RUF012


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class RunStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class ModelStage(enum.StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    datasets: Mapped[list[Dataset]] = relationship(back_populates="project")
    jobs: Mapped[list[Job]] = relationship(back_populates="project")
    models: Mapped[list[Model]] = relationship(back_populates="project")
    experiments: Mapped[list[Experiment]] = relationship(back_populates="project")


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("project_id", "slug", name="uq_dataset_slug"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    modality: Mapped[Modality] = mapped_column(Enum(Modality))
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(500))
    license_default: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    project: Mapped[Project] = relationship(back_populates="datasets")
    versions: Mapped[list[DatasetVersion]] = relationship(back_populates="dataset")


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    version: Mapped[str] = mapped_column(String(32))
    manifest_uri: Mapped[str] = mapped_column(String(1000))
    num_samples: Mapped[int] = mapped_column(Integer, default=0)
    total_audio_s: Mapped[float] = mapped_column(Float, default=0.0)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[str | None] = mapped_column(String(200))
    # ON DELETE SET NULL — a fork must survive deletion of its ancestor; this
    # column captures provenance, it is not a structural dependency.
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="SET NULL"), index=True, nullable=True
    )

    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    parent: Mapped[DatasetVersion | None] = relationship(
        "DatasetVersion", remote_side="DatasetVersion.id", foreign_keys=[parent_version_id]
    )


class Job(Base):
    """A unit of work submitted to the queue (train, eval, export, import...)."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.QUEUED, index=True
    )
    priority: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    project: Mapped[Project] = relationship(back_populates="jobs")
    runs: Mapped[list[Run]] = relationship(back_populates="job")


class Run(Base):
    """A single attempt at executing a Job. Jobs may have multiple Runs (retries)."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), default=RunStatus.PENDING, index=True
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    logs_uri: Mapped[str | None] = mapped_column(String(1000))
    artifacts_uri: Mapped[str | None] = mapped_column(String(1000))
    code_commit: Mapped[str | None] = mapped_column(String(64))
    image_digest: Mapped[str | None] = mapped_column(String(120))
    hardware: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    seed: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="runs")


class Experiment(Base):
    """A named grouping of runs for comparison."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    run_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="experiments")


class Model(Base):
    """A logical model in the registry. Holds many versions."""

    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("project_id", "slug", name="uq_model_slug"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    modality: Mapped[Modality] = mapped_column(Enum(Modality))
    family: Mapped[str | None] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="models")
    versions: Mapped[list[ModelVersion]] = relationship(back_populates="model")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_model_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), index=True)
    version: Mapped[str] = mapped_column(String(32))
    stage: Mapped[ModelStage] = mapped_column(Enum(ModelStage), default=ModelStage.DEV)
    artifact_uri: Mapped[str] = mapped_column(String(1000))
    format: Mapped[str] = mapped_column(String(64), default="safetensors")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"))
    source_dataset_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id")
    )
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    model: Mapped[Model] = relationship(back_populates="versions")


class S2SPipeline(Base):
    """A pipeline-mode speech-to-speech assistant: ASR + LLM + TTS + policy."""

    __tablename__ = "s2s_pipelines"
    __table_args__ = (UniqueConstraint("project_id", "slug", name="uq_s2s_slug"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)

    asr_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"))
    llm_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"))
    tts_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_versions.id"))

    # Fallback configs when no in-registry model is selected (use HF id directly).
    asr_fallback: Mapped[str | None] = mapped_column(String(200))
    llm_fallback: Mapped[str | None] = mapped_column(String(200))
    tts_fallback: Mapped[str | None] = mapped_column(String(200))

    system_prompt: Mapped[str | None] = mapped_column(Text)
    vad_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    runtime_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class KernelDraft(Base):
    """A custom CUDA / Triton / Pallas kernel under development.

    Stores source code, a reference (numpy / torch) snippet for correctness
    checks, and the latest benchmark results so the Lab UI can show progress.
    """

    __tablename__ = "kernel_drafts"
    __table_args__ = (UniqueConstraint("project_id", "slug", name="uq_kernel_slug"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    backend: Mapped[str] = mapped_column(String(32))
    op: Mapped[str] = mapped_column(String(64), default="custom")
    source: Mapped[str] = mapped_column(Text)
    reference: Mapped[str | None] = mapped_column(Text)
    bench_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_bench: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


# Add native-mode column to S2SPipeline via a follow-on table alteration is
# overkill in v1; instead we store mode + per-mode config inside runtime_config.


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(255))
    is_superuser: Mapped[bool] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Role(enum.StrEnum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uq_membership"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.EDITOR)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    """Immutable record of a privileged action.

    Written by the audit middleware on every mutating request that lands a
    2xx response. Designed to be append-only — there is no UPDATE route.
    """

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    actor_user_id: Mapped[str | None] = mapped_column(String(32), index=True)
    actor_email: Mapped[str] = mapped_column(String(320))
    method: Mapped[str] = mapped_column(String(8))
    path: Mapped[str] = mapped_column(String(500), index=True)
    project_id: Mapped[str | None] = mapped_column(String(32), index=True)
    status_code: Mapped[int] = mapped_column(Integer)
    body_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    request_id: Mapped[str | None] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class ModelSpec(Base):
    """Persisted architecture specification for from-scratch pretraining.

    The `spec` JSON describes blocks, dimensions, vocab, and per-modality
    options. `oas_core.architectures.factory.build_from_spec` materializes it
    into a torch.nn.Module that the corresponding pretrain handler consumes.
    """

    __tablename__ = "model_specs"
    __table_args__ = (UniqueConstraint("project_id", "slug", name="uq_model_spec_slug"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    slug: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    modality: Mapped[Modality] = mapped_column(Enum(Modality))
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    estimated_params: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
