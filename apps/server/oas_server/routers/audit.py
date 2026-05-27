"""Read-only audit log endpoint (superuser only)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from oas_core.db import AuditLog, Role, session_scope
from pydantic import BaseModel
from sqlalchemy import select

from oas_server.auth import CurrentUser, require_user, assert_role

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditOut(BaseModel):
    id: str
    actor_user_id: str | None
    actor_email: str
    method: str
    path: str
    project_id: str | None
    status_code: int
    body_summary: dict[str, Any]
    request_id: str | None
    created_at: datetime


@router.get("", response_model=list[AuditOut])
def list_audit(
    limit: int = Query(default=100, le=500),
    project_id: str | None = None,
    path_prefix: str | None = None,
    user: CurrentUser = Depends(require_user),
) -> list[AuditOut]:
    if not user.is_superuser:
        if not project_id:
            raise HTTPException(403, "Only superusers can view all audit logs. Specify a project_id to view scoped project logs.")
        assert_role(user, project_id, Role.VIEWER)
    with session_scope() as s:
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        if project_id:
            stmt = stmt.where(AuditLog.project_id == project_id)
        if path_prefix:
            stmt = stmt.where(AuditLog.path.like(f"{path_prefix}%"))
        return [
            AuditOut(
                id=a.id,
                actor_user_id=a.actor_user_id,
                actor_email=a.actor_email,
                method=a.method,
                path=a.path,
                project_id=a.project_id,
                status_code=a.status_code,
                body_summary=a.body_summary or {},
                request_id=a.request_id,
                created_at=a.created_at,
            )
            for a in s.scalars(stmt)
        ]
