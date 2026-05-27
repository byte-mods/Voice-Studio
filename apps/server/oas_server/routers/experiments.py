from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from oas_core.db import Experiment, Project, session_scope
from pydantic import BaseModel, Field
from sqlalchemy import select

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentIn(BaseModel):
    project_id: str
    name: str
    description: str | None = None
    run_ids: list[str] = Field(default_factory=list)


class ExperimentOut(BaseModel):
    id: str
    project_id: str
    name: str
    description: str | None
    run_ids: list[str]
    created_at: datetime


def _to_out(e: Experiment) -> ExperimentOut:
    return ExperimentOut(
        id=e.id,
        project_id=e.project_id,
        name=e.name,
        description=e.description,
        run_ids=e.run_ids or [],
        created_at=e.created_at,
    )


@router.get("", response_model=list[ExperimentOut])
def list_experiments(project_id: str | None = None) -> list[ExperimentOut]:
    with session_scope() as s:
        stmt = select(Experiment)
        if project_id:
            stmt = stmt.where(Experiment.project_id == project_id)
        return [_to_out(e) for e in s.scalars(stmt.order_by(Experiment.created_at.desc()))]


@router.post("", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
def create_experiment(body: ExperimentIn) -> ExperimentOut:
    with session_scope() as s:
        if not s.get(Project, body.project_id):
            raise HTTPException(404, "project not found")
        e = Experiment(
            project_id=body.project_id,
            name=body.name,
            description=body.description,
            run_ids=body.run_ids,
        )
        s.add(e)
        s.flush()
        return _to_out(e)
