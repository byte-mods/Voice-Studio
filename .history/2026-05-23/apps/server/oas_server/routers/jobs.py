from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from oas_core.db import Job, JobStatus, Role, Run, RunStatus, session_scope
from oas_core.queue import submit_job
from oas_core.queue.backend import list_handlers
from pydantic import BaseModel, Field
from sqlalchemy import select

from oas_server.auth import CurrentUser, assert_role, require_user

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobIn(BaseModel):
    project_id: str
    kind: str
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class RunOut(BaseModel):
    id: str
    job_id: str
    attempt: int
    status: RunStatus
    metrics: dict[str, Any]
    logs_uri: str | None
    artifacts_uri: str | None
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None


class JobOut(BaseModel):
    id: str
    project_id: str
    kind: str
    name: str
    config: dict[str, Any]
    status: JobStatus
    priority: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    runs: list[RunOut] = Field(default_factory=list)


def _to_run_out(r: Run) -> RunOut:
    return RunOut(
        id=r.id,
        job_id=r.job_id,
        attempt=r.attempt,
        status=r.status,
        metrics=r.metrics or {},
        logs_uri=r.logs_uri,
        artifacts_uri=r.artifacts_uri,
        started_at=r.started_at,
        finished_at=r.finished_at,
        error=r.error,
    )


def _to_job_out(j: Job, runs: list[Run] | None = None) -> JobOut:
    return JobOut(
        id=j.id,
        project_id=j.project_id,
        kind=j.kind,
        name=j.name,
        config=j.config or {},
        status=j.status,
        priority=j.priority,
        error=j.error,
        created_at=j.created_at,
        started_at=j.started_at,
        finished_at=j.finished_at,
        runs=[_to_run_out(r) for r in (runs or [])],
    )


@router.get("/handlers", response_model=list[str])
def list_kinds() -> list[str]:
    return list_handlers()


@router.get("", response_model=list[JobOut])
def list_jobs(
    project_id: str | None = None,
    status_filter: JobStatus | None = None,
    limit: int = 100,
) -> list[JobOut]:
    with session_scope() as s:
        stmt = select(Job)
        if project_id:
            stmt = stmt.where(Job.project_id == project_id)
        if status_filter:
            stmt = stmt.where(Job.status == status_filter)
        stmt = stmt.order_by(Job.created_at.desc()).limit(limit)
        return [_to_job_out(j) for j in s.scalars(stmt)]


@router.post("", response_model=JobOut, status_code=status.HTTP_201_CREATED)
def create_job(body: JobIn, user: CurrentUser = Depends(require_user)) -> JobOut:
    if body.kind not in list_handlers():
        raise HTTPException(400, f"Unknown job kind {body.kind!r}. Known: {list_handlers()}")
    assert_role(user, body.project_id, Role.EDITOR)
    job_id = submit_job(body.project_id, body.kind, body.name, body.config, priority=body.priority)
    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job is not None
        return _to_job_out(job)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if not job:
            raise HTTPException(404)
        runs = list(s.scalars(select(Run).where(Run.job_id == job_id).order_by(Run.attempt)))
        return _to_job_out(job, runs)


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(
    job_id: str, request: Request, user: CurrentUser = Depends(require_user)
) -> JobOut:
    with session_scope() as s:
        job = s.get(Job, job_id)
        if not job:
            raise HTTPException(404)
        assert_role(user, job.project_id, Role.EDITOR)
    pool = getattr(request.app.state, "worker_pool", None)
    if pool is not None:
        pool.cancel(job_id)
    with session_scope() as s:
        job = s.get(Job, job_id)
        if not job:
            raise HTTPException(404)
        return _to_job_out(job)
