"""Model registry service.

Thin functional API on top of the SQLAlchemy models. UI and SDK both call
through this layer so authorization, validation, and side-effects live in
exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from oas_core.db import Model, ModelStage, ModelVersion, session_scope
from oas_core.db.models import Modality


@dataclass(frozen=True, slots=True)
class ModelLineage:
    model_id: str
    version: str
    source_run_id: str | None
    source_dataset_version_id: str | None


def create_model(
    project_id: str,
    slug: str,
    name: str,
    modality: Modality,
    *,
    family: str | None = None,
    description: str | None = None,
) -> str:
    with session_scope() as s:
        m = Model(
            project_id=project_id,
            slug=slug,
            name=name,
            modality=modality,
            family=family,
            description=description,
        )
        s.add(m)
        s.flush()
        return m.id


def get_model(model_id: str) -> Model | None:
    with session_scope() as s:
        return s.get(Model, model_id)


def list_models(project_id: str, modality: Modality | None = None) -> list[Model]:
    with session_scope() as s:
        stmt = select(Model).where(Model.project_id == project_id)
        if modality is not None:
            stmt = stmt.where(Model.modality == modality)
        return list(s.scalars(stmt).all())


def publish_version(
    model_id: str,
    version: str,
    artifact_uri: str,
    *,
    format: str = "safetensors",
    size_bytes: int = 0,
    sha256: str | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    source_run_id: str | None = None,
    source_dataset_version_id: str | None = None,
    stage: ModelStage = ModelStage.DEV,
    notes: str | None = None,
) -> str:
    with session_scope() as s:
        v = ModelVersion(
            model_id=model_id,
            version=version,
            artifact_uri=artifact_uri,
            format=format,
            size_bytes=size_bytes,
            sha256=sha256,
            config=config or {},
            metrics=metrics or {},
            source_run_id=source_run_id,
            source_dataset_version_id=source_dataset_version_id,
            stage=stage,
            notes=notes,
        )
        s.add(v)
        s.flush()
        return v.id


def get_version(version_id: str) -> ModelVersion | None:
    with session_scope() as s:
        return s.get(ModelVersion, version_id)


def list_versions(model_id: str) -> list[ModelVersion]:
    with session_scope() as s:
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_id == model_id)
            .order_by(ModelVersion.created_at.desc())
        )
        return list(s.scalars(stmt).all())


def set_stage(version_id: str, stage: ModelStage) -> None:
    with session_scope() as s:
        v = s.get(ModelVersion, version_id)
        if v is None:
            raise KeyError(f"ModelVersion {version_id!r} not found")
        v.stage = stage
