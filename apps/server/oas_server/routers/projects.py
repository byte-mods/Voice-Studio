from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from oas_core.db import Membership, Project, Role, session_scope
from pydantic import BaseModel, Field
from sqlalchemy import select

from oas_server.auth import CurrentUser, assert_role, require_user

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectIn(BaseModel):
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-_]*$")
    name: str
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class ProjectOut(BaseModel):
    id: str
    slug: str
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    settings: dict[str, Any]


def _redact_secrets(d: Any) -> Any:
    if isinstance(d, dict):
        res = {}
        for k, v in d.items():
            k_lower = k.lower()
            if any(term in k_lower for term in ("key", "secret", "password", "token")) and not (k_lower.endswith("_id") or k_lower.endswith("id")):
                res[k] = "********"
            else:
                res[k] = _redact_secrets(v)
        return res
    elif isinstance(d, list):
        return [_redact_secrets(x) for x in d]
    return d


def _to_out(p: Project) -> ProjectOut:
    return ProjectOut(
        id=p.id,
        slug=p.slug,
        name=p.name,
        description=p.description,
        created_at=p.created_at,
        updated_at=p.updated_at,
        settings=_redact_secrets(p.settings or {}),
    )


@router.get("", response_model=list[ProjectOut])
def list_projects() -> list[ProjectOut]:
    with session_scope() as s:
        return [_to_out(p) for p in s.scalars(select(Project).order_by(Project.created_at.desc()))]


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectIn, user: CurrentUser = Depends(require_user)) -> ProjectOut:
    with session_scope() as s:
        existing = s.scalar(select(Project).where(Project.slug == body.slug))
        if existing:
            raise HTTPException(409, f"Project slug {body.slug!r} already exists")
        p = Project(slug=body.slug, name=body.name, description=body.description, settings=body.settings)
        s.add(p)
        s.flush()
        # Creator automatically becomes admin (unless anonymous bootstrap).
        if not user.anonymous:
            s.add(Membership(project_id=p.id, user_id=user.id, role=Role.ADMIN))
        return _to_out(p)


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str) -> ProjectOut:
    with session_scope() as s:
        p = s.get(Project, project_id)
        if not p:
            raise HTTPException(404)
        return _to_out(p)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, user: CurrentUser = Depends(require_user)) -> None:
    assert_role(user, project_id, Role.ADMIN)
    with session_scope() as s:
        p = s.get(Project, project_id)
        if not p:
            raise HTTPException(404)
        s.delete(p)


class ProjectUpdateIn(BaseModel):
    name: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None


@router.put("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: str,
    body: ProjectUpdateIn,
    user: CurrentUser = Depends(require_user),
) -> ProjectOut:
    assert_role(user, project_id, Role.ADMIN)
    with session_scope() as s:
        p = s.get(Project, project_id)
        if not p:
            raise HTTPException(404)
        if body.name is not None:
            p.name = body.name
        if body.description is not None:
            p.description = body.description
        if body.settings is not None:
            current = dict(p.settings or {})
            for k, v in body.settings.items():
                if v == "********":
                    pass
                else:
                    current[k] = v
            p.settings = current
        s.flush()
        return _to_out(p)
