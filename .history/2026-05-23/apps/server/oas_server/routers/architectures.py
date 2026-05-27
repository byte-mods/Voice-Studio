"""Architecture (ModelSpec) CRUD + pretrain dispatch."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from oas_core.architectures import estimate_params, validate_spec
from oas_core.db import ModelSpec, Project, Role, session_scope
from oas_core.db.models import Modality
from oas_core.queue import submit_job
from pydantic import BaseModel, Field
from sqlalchemy import select

from oas_server.auth import CurrentUser, assert_role, require_user

router = APIRouter(prefix="/architectures", tags=["architectures"])


class ArchitectureIn(BaseModel):
    project_id: str
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-_.]*$")
    name: str
    modality: Modality
    spec: dict[str, Any]
    notes: str | None = None


class ArchitectureOut(BaseModel):
    id: str
    project_id: str
    slug: str
    name: str
    modality: Modality
    spec: dict[str, Any]
    estimated_params: int
    notes: str | None
    created_at: datetime
    updated_at: datetime


def _to_out(m: ModelSpec) -> ArchitectureOut:
    return ArchitectureOut(
        id=m.id,
        project_id=m.project_id,
        slug=m.slug,
        name=m.name,
        modality=m.modality,
        spec=m.spec or {},
        estimated_params=m.estimated_params,
        notes=m.notes,
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


@router.post("/validate")
def validate_endpoint(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_spec(spec)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "estimated_params": estimate_params(spec)}


@router.get("", response_model=list[ArchitectureOut])
def list_architectures(project_id: str | None = None) -> list[ArchitectureOut]:
    with session_scope() as s:
        stmt = select(ModelSpec)
        if project_id:
            stmt = stmt.where(ModelSpec.project_id == project_id)
        stmt = stmt.order_by(ModelSpec.updated_at.desc())
        return [_to_out(m) for m in s.scalars(stmt)]


@router.post("", response_model=ArchitectureOut, status_code=status.HTTP_201_CREATED)
def create_architecture(
    body: ArchitectureIn, user: CurrentUser = Depends(require_user)
) -> ArchitectureOut:
    assert_role(user, body.project_id, Role.EDITOR)
    try:
        validate_spec(body.spec)
        n = estimate_params(body.spec)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    with session_scope() as s:
        if not s.get(Project, body.project_id):
            raise HTTPException(404, "project not found")
        m = ModelSpec(
            project_id=body.project_id,
            slug=body.slug,
            name=body.name,
            modality=body.modality,
            spec=body.spec,
            estimated_params=n,
            notes=body.notes,
        )
        s.add(m)
        s.flush()
        return _to_out(m)


@router.get("/{spec_id}", response_model=ArchitectureOut)
def get_architecture(spec_id: str) -> ArchitectureOut:
    with session_scope() as s:
        m = s.get(ModelSpec, spec_id)
        if not m:
            raise HTTPException(404)
        return _to_out(m)


class PatchIn(BaseModel):
    name: str | None = None
    spec: dict[str, Any] | None = None
    notes: str | None = None


@router.patch("/{spec_id}", response_model=ArchitectureOut)
def patch_architecture(
    spec_id: str, body: PatchIn, user: CurrentUser = Depends(require_user)
) -> ArchitectureOut:
    with session_scope() as s:
        m = s.get(ModelSpec, spec_id)
        if not m:
            raise HTTPException(404)
        assert_role(user, m.project_id, Role.EDITOR)
        if body.name is not None:
            m.name = body.name
        if body.notes is not None:
            m.notes = body.notes
        if body.spec is not None:
            try:
                validate_spec(body.spec)
                m.estimated_params = estimate_params(body.spec)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            m.spec = body.spec
        s.flush()
        return _to_out(m)


class PretrainIn(BaseModel):
    dataset_version_id: str
    training: dict[str, Any] = Field(default_factory=dict)
    publish_model_slug: str | None = None
    publish_version: str = "0.1.0"


@router.post("/{spec_id}/pretrain")
def pretrain(
    spec_id: str, body: PretrainIn, user: CurrentUser = Depends(require_user)
) -> dict[str, str]:
    with session_scope() as s:
        m = s.get(ModelSpec, spec_id)
        if not m:
            raise HTTPException(404)
        assert_role(user, m.project_id, Role.EDITOR)
        modality = m.modality
        project_id = m.project_id

    kind = {
        Modality.LLM: "llm_pretrain",
        Modality.ASR: "asr_pretrain",
        Modality.TTS: "tts_pretrain",
    }.get(modality)
    if not kind:
        raise HTTPException(400, f"pretrain not supported for modality {modality}")

    config: dict[str, Any] = {
        "model_spec_id": spec_id,
        "dataset_version_id": body.dataset_version_id,
        "training": body.training,
    }

    if body.publish_model_slug:
        from oas_core.db import Model

        with session_scope() as s:
            existing = (
                s.query(Model)
                .filter(Model.project_id == project_id, Model.slug == body.publish_model_slug)
                .first()
            )
            if existing:
                model_id = existing.id
            else:
                new_model = Model(
                    project_id=project_id,
                    slug=body.publish_model_slug,
                    name=body.publish_model_slug,
                    modality=modality,
                )
                s.add(new_model)
                s.flush()
                model_id = new_model.id
        config["registry"] = {"model_id": model_id, "version": body.publish_version}

    job_id = submit_job(project_id, kind, f"pretrain {modality.value}", config)
    return {"job_id": job_id}
