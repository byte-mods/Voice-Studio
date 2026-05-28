from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from oas_core.db import Job, JobStatus, Model, Run, RunStatus, session_scope
from oas_core.db.models import Modality, ModelStage, Role
from oas_core.db.models import ModelVersion as ORMModelVersion
from oas_core.registry import (
    create_model,
    get_model,
    list_models,
    list_versions,
    publish_version,
    set_stage,
)
from pydantic import BaseModel, Field
from sqlalchemy import select

from oas_server.auth import (
    _ROLE_RANK,
    CurrentUser,
    _role_for,
    assert_role,
    require_user,
)

router = APIRouter(prefix="/models", tags=["models"])


class ModelIn(BaseModel):
    project_id: str
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-_.]*$")
    name: str
    modality: Modality
    family: str | None = None
    description: str | None = None


class ModelOut(BaseModel):
    id: str
    project_id: str
    slug: str
    name: str
    modality: Modality
    family: str | None
    description: str | None
    created_at: datetime


class VersionIn(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    artifact_uri: str
    format: str = "safetensors"
    size_bytes: int = 0
    sha256: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    source_run_id: str | None = None
    source_dataset_version_id: str | None = None
    notes: str | None = None


class VersionOut(BaseModel):
    id: str
    model_id: str
    version: str
    stage: ModelStage
    artifact_uri: str
    format: str
    size_bytes: int
    sha256: str | None
    config: dict[str, Any]
    metrics: dict[str, Any]
    source_run_id: str | None
    source_dataset_version_id: str | None
    notes: str | None
    created_at: datetime


@router.get("", response_model=list[ModelOut])
def list_models_endpoint(project_id: str, modality: Modality | None = None) -> list[ModelOut]:
    return [
        ModelOut(
            id=m.id,
            project_id=m.project_id,
            slug=m.slug,
            name=m.name,
            modality=m.modality,
            family=m.family,
            description=m.description,
            created_at=m.created_at,
        )
        for m in list_models(project_id, modality)
    ]


@router.post("", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
def create_model_endpoint(body: ModelIn, user: CurrentUser = Depends(require_user)) -> ModelOut:
    assert_role(user, body.project_id, Role.EDITOR)
    mid = create_model(
        body.project_id,
        body.slug,
        body.name,
        body.modality,
        family=body.family,
        description=body.description,
    )
    from oas_core.registry import get_model

    m = get_model(mid)
    assert m
    return ModelOut(
        id=m.id,
        project_id=m.project_id,
        slug=m.slug,
        name=m.name,
        modality=m.modality,
        family=m.family,
        description=m.description,
        created_at=m.created_at,
    )


@router.get("/{model_id}/versions", response_model=list[VersionOut])
def list_versions_endpoint(model_id: str) -> list[VersionOut]:
    return [_v_out(v) for v in list_versions(model_id)]


@router.post("/{model_id}/versions", response_model=VersionOut, status_code=201)
def publish_version_endpoint(
    model_id: str, body: VersionIn, user: CurrentUser = Depends(require_user)
) -> VersionOut:
    with session_scope() as s:
        m = s.get(Model, model_id)
        if not m:
            raise HTTPException(404)
        assert_role(user, m.project_id, Role.EDITOR)
    vid = publish_version(
        model_id,
        body.version,
        body.artifact_uri,
        format=body.format,
        size_bytes=body.size_bytes,
        sha256=body.sha256,
        config=body.config,
        metrics=body.metrics,
        source_run_id=body.source_run_id,
        source_dataset_version_id=body.source_dataset_version_id,
        notes=body.notes,
    )
    from oas_core.registry import get_version

    v = get_version(vid)
    assert v
    return _v_out(v)


class HFModelSearchResult(BaseModel):
    id: str
    downloads: int
    likes: int
    pipeline_tag: str | None
    tags: list[str]
    last_modified: str | None


@router.get("/hf/search", response_model=list[HFModelSearchResult])
def search_hf_models(
    query: str | None = None,
    modality: Modality | None = None,
) -> list[HFModelSearchResult]:
    from huggingface_hub import HfApi

    api = HfApi()
    filter_tag = None
    if modality:
        if modality == Modality.ASR:
            filter_tag = "automatic-speech-recognition"
        elif modality == Modality.TTS:
            filter_tag = "text-to-speech"
        elif modality == Modality.LLM:
            filter_tag = "text-generation"
        elif modality == Modality.S2S:
            # S2S can be audio-to-audio
            filter_tag = "audio-to-audio"

    try:
        models = api.list_models(
            search=query or None,
            filter=filter_tag,
            sort="downloads",
            direction=-1,
            limit=20,
        )

        results = []
        for m in models:
            lm_str = None
            if getattr(m, "last_modified", None):
                lm_str = m.last_modified.isoformat() if hasattr(m.last_modified, "isoformat") else str(m.last_modified)
            results.append(
                HFModelSearchResult(
                    id=m.id,
                    downloads=getattr(m, "downloads", 0) or 0,
                    likes=getattr(m, "likes", 0) or 0,
                    pipeline_tag=getattr(m, "pipeline_tag", None),
                    tags=getattr(m, "tags", []) or [],
                    last_modified=lm_str,
                )
            )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EvalResult(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: datetime
    finished_at: datetime | None
    metrics: dict[str, Any]


@router.get("/{model_id}", response_model=ModelOut)
def get_model_endpoint(model_id: str) -> ModelOut:
    m = get_model(model_id)
    if not m:
        raise HTTPException(404)
    return ModelOut(
        id=m.id,
        project_id=m.project_id,
        slug=m.slug,
        name=m.name,
        modality=m.modality,
        family=m.family,
        description=m.description,
        created_at=m.created_at,
    )


@router.get("/versions/{version_id}/evals", response_model=list[EvalResult])
def list_evals(version_id: str) -> list[EvalResult]:
    """Return every completed eval job whose config targeted this ModelVersion.

    Eval handlers (`asr_eval`, `llm_eval`, `tts_eval`) all read
    `model_version_id` from their config, so we filter the `jobs` table on
    that key.
    """
    eval_kinds = ("asr_eval", "llm_eval", "tts_eval")
    out: list[EvalResult] = []
    with session_scope() as s:
        stmt = (
            select(Job)
            .where(Job.kind.in_(eval_kinds))
            .order_by(Job.created_at.desc())
        )
        for job in s.scalars(stmt):
            if (job.config or {}).get("model_version_id") != version_id:
                continue
            metrics: dict[str, Any] = {}
            if job.status == JobStatus.SUCCEEDED:
                latest_run = (
                    s.query(Run)
                    .filter(Run.job_id == job.id, Run.status == RunStatus.SUCCEEDED)
                    .order_by(Run.attempt.desc())
                    .first()
                )
                if latest_run is not None:
                    metrics = dict(latest_run.metrics or {})
            out.append(
                EvalResult(
                    job_id=job.id,
                    kind=job.kind,
                    status=job.status.value,
                    created_at=job.created_at,
                    finished_at=job.finished_at,
                    metrics=metrics,
                )
            )
    return out


@router.post("/versions/{version_id}/stage", response_model=VersionOut)
def set_stage_endpoint(
    version_id: str,
    stage: ModelStage,
    user: CurrentUser = Depends(require_user),
) -> VersionOut:
    # Stage promotion is privileged: require admin on the owning project.
    with session_scope() as s:
        v = s.get(ORMModelVersion, version_id)
        if not v:
            raise HTTPException(404, "version not found")
        m = s.get(Model, v.model_id)
        if not m:
            raise HTTPException(404, "model not found")
        role = _role_for(user, m.project_id)
        if role is None or _ROLE_RANK[role] < _ROLE_RANK[Role.ADMIN]:
            raise HTTPException(
                403, "promoting a model version requires admin role on the project"
            )
    try:
        set_stage(version_id, stage)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    from oas_core.registry import get_version

    v = get_version(version_id)
    assert v
    return _v_out(v)


class MOSRatingIn(BaseModel):
    score: float = Field(..., ge=1.0, le=5.0)


@router.post("/versions/{version_id}/mos", response_model=VersionOut)
def record_mos_endpoint(
    version_id: str,
    body: MOSRatingIn,
    user: CurrentUser = Depends(require_user),
) -> VersionOut:
    with session_scope() as s:
        v = s.get(ORMModelVersion, version_id)
        if not v:
            raise HTTPException(404, "version not found")
        metrics = dict(v.metrics or {})
        ratings = list(metrics.get("mos_ratings", []))
        ratings.append(body.score)
        metrics["mos_ratings"] = ratings
        metrics["mos"] = round(sum(ratings) / len(ratings), 2)
        v.metrics = metrics
        s.commit()
        # Return updated version out
        return _v_out(v)



def _v_out(v: Any) -> VersionOut:
    return VersionOut(
        id=v.id,
        model_id=v.model_id,
        version=v.version,
        stage=v.stage,
        artifact_uri=v.artifact_uri,
        format=v.format,
        size_bytes=v.size_bytes,
        sha256=v.sha256,
        config=v.config or {},
        metrics=v.metrics or {},
        source_run_id=v.source_run_id,
        source_dataset_version_id=v.source_dataset_version_id,
        notes=v.notes,
        created_at=v.created_at,
    )
