"""Architecture Lab: custom kernel drafts (Triton / CUDA / Pallas) + benchmarking."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from oas_core.db import KernelDraft, Project, Role, session_scope
from oas_core.queue import submit_job
from pydantic import BaseModel, Field
from sqlalchemy import select

from oas_server.auth import CurrentUser, assert_role, require_user

router = APIRouter(prefix="/kernels", tags=["kernels"])


class KernelIn(BaseModel):
    project_id: str
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-_.]*$")
    name: str
    backend: str = Field(pattern=r"^(triton|cuda|pallas)$")
    op: str = "custom"
    source: str
    reference: str | None = None
    bench_config: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class KernelUpdate(BaseModel):
    name: str | None = None
    source: str | None = None
    reference: str | None = None
    bench_config: dict[str, Any] | None = None
    notes: str | None = None


class KernelOut(BaseModel):
    id: str
    project_id: str
    slug: str
    name: str
    backend: str
    op: str
    source: str
    reference: str | None
    bench_config: dict[str, Any]
    last_bench: dict[str, Any]
    notes: str | None
    created_at: datetime
    updated_at: datetime


def _to_out(k: KernelDraft) -> KernelOut:
    return KernelOut(
        id=k.id,
        project_id=k.project_id,
        slug=k.slug,
        name=k.name,
        backend=k.backend,
        op=k.op,
        source=k.source,
        reference=k.reference,
        bench_config=k.bench_config or {},
        last_bench=k.last_bench or {},
        notes=k.notes,
        created_at=k.created_at,
        updated_at=k.updated_at,
    )


@router.get("", response_model=list[KernelOut])
def list_kernels(project_id: str | None = None) -> list[KernelOut]:
    with session_scope() as s:
        stmt = select(KernelDraft)
        if project_id:
            stmt = stmt.where(KernelDraft.project_id == project_id)
        stmt = stmt.order_by(KernelDraft.updated_at.desc())
        return [_to_out(k) for k in s.scalars(stmt)]


@router.post("", response_model=KernelOut, status_code=status.HTTP_201_CREATED)
def create_kernel(body: KernelIn, user: CurrentUser = Depends(require_user)) -> KernelOut:
    assert_role(user, body.project_id, Role.EDITOR)
    with session_scope() as s:
        if not s.get(Project, body.project_id):
            raise HTTPException(404, "project not found")
        k = KernelDraft(
            project_id=body.project_id,
            slug=body.slug,
            name=body.name,
            backend=body.backend,
            op=body.op,
            source=body.source,
            reference=body.reference,
            bench_config=body.bench_config,
            notes=body.notes,
        )
        s.add(k)
        s.flush()
        return _to_out(k)


@router.get("/{kernel_id}", response_model=KernelOut)
def get_kernel(kernel_id: str) -> KernelOut:
    with session_scope() as s:
        k = s.get(KernelDraft, kernel_id)
        if not k:
            raise HTTPException(404)
        return _to_out(k)


@router.patch("/{kernel_id}", response_model=KernelOut)
def update_kernel(
    kernel_id: str, patch: KernelUpdate, user: CurrentUser = Depends(require_user)
) -> KernelOut:
    with session_scope() as s:
        k = s.get(KernelDraft, kernel_id)
        if not k:
            raise HTTPException(404)
        assert_role(user, k.project_id, Role.EDITOR)
        payload = patch.model_dump(exclude_unset=True)
        for key, val in payload.items():
            setattr(k, key, val)
        s.flush()
        return _to_out(k)


@router.delete("/{kernel_id}", status_code=204)
def delete_kernel(kernel_id: str, user: CurrentUser = Depends(require_user)) -> None:
    with session_scope() as s:
        k = s.get(KernelDraft, kernel_id)
        if not k:
            raise HTTPException(404)
        assert_role(user, k.project_id, Role.ADMIN)
        s.delete(k)


class BenchmarkIn(BaseModel):
    autotune_grid: dict[str, list[Any]] = Field(default_factory=dict)
    profile: bool = False


@router.post("/{kernel_id}/benchmark")
def benchmark(
    kernel_id: str,
    body: BenchmarkIn | None = None,
    user: CurrentUser = Depends(require_user),
) -> dict[str, str]:
    """Queue a kernel_bench job for this kernel; returns the job id."""
    body = body or BenchmarkIn()
    with session_scope() as s:
        k = s.get(KernelDraft, kernel_id)
        if not k:
            raise HTTPException(404)
        assert_role(user, k.project_id, Role.EDITOR)
        config: dict[str, Any] = {"kernel_id": k.id}
        if body.autotune_grid:
            config["autotune_grid"] = body.autotune_grid
        if body.profile:
            config["profile"] = True
        job_id = submit_job(k.project_id, "kernel_bench", f"bench {k.slug}", config)
        return {"job_id": job_id}
