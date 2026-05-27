"""Auth endpoints: signup, login, me, project membership management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from oas_core.db import Membership, Project, Role, User, session_scope
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from oas_server.auth import (
    CurrentUser,
    hash_password,
    issue_token,
    require_user,
    verify_password,
    assert_role,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class SignupIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str | None = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class UserOut(BaseModel):
    id: str
    email: str
    name: str | None
    is_superuser: bool
    created_at: datetime | None = None


@router.post("/signup", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def signup(body: SignupIn) -> TokenOut:
    with session_scope() as s:
        if s.scalar(select(User).where(User.email == body.email)):
            raise HTTPException(409, "email already in use")
        # First-ever signup becomes superuser.
        first = s.scalar(select(User).limit(1)) is None
        u = User(
            email=body.email,
            name=body.name,
            password_hash=hash_password(body.password),
            is_superuser=1 if first else 0,
        )
        s.add(u)
        s.flush()
        token = issue_token(u.id)
        return TokenOut(
            access_token=token,
            user={"id": u.id, "email": u.email, "name": u.name, "is_superuser": bool(u.is_superuser)},
        )


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn) -> TokenOut:
    with session_scope() as s:
        u = s.scalar(select(User).where(User.email == body.email))
        if not u or not verify_password(body.password, u.password_hash):
            raise HTTPException(401, "invalid credentials")
        token = issue_token(u.id)
        return TokenOut(
            access_token=token,
            user={"id": u.id, "email": u.email, "name": u.name, "is_superuser": bool(u.is_superuser)},
        )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser = Depends(require_user)) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        name=user.name,
        is_superuser=user.is_superuser,
    )


class MembershipIn(BaseModel):
    user_id: str
    role: Role = Role.EDITOR


class MembershipOut(BaseModel):
    id: str
    project_id: str
    user_id: str
    email: str
    role: Role


@router.get("/projects/{project_id}/members", response_model=list[MembershipOut])
def list_members(project_id: str, user: CurrentUser = Depends(require_user)) -> list[MembershipOut]:
    assert_role(user, project_id, Role.VIEWER)
    with session_scope() as s:
        if not s.get(Project, project_id):
            raise HTTPException(404)
        out: list[MembershipOut] = []
        for m in s.scalars(select(Membership).where(Membership.project_id == project_id)):
            u = s.get(User, m.user_id)
            if u:
                out.append(
                    MembershipOut(id=m.id, project_id=m.project_id, user_id=u.id, email=u.email, role=m.role)
                )
        return out


@router.post("/projects/{project_id}/members", response_model=MembershipOut, status_code=201)
def add_member(
    project_id: str, body: MembershipIn, user: CurrentUser = Depends(require_user)
) -> MembershipOut:
    assert_role(user, project_id, Role.ADMIN)
    with session_scope() as s:
        if not s.get(Project, project_id):
            raise HTTPException(404, "project not found")
        target = s.get(User, body.user_id)
        if not target:
            raise HTTPException(404, "user not found")
        existing = s.scalar(
            select(Membership).where(
                Membership.project_id == project_id, Membership.user_id == body.user_id
            )
        )
        if existing:
            existing.role = body.role
            return MembershipOut(
                id=existing.id, project_id=project_id, user_id=target.id, email=target.email, role=body.role
            )
        m = Membership(project_id=project_id, user_id=body.user_id, role=body.role)
        s.add(m)
        s.flush()
        return MembershipOut(id=m.id, project_id=project_id, user_id=target.id, email=target.email, role=body.role)


@router.delete("/projects/{project_id}/members/{membership_id}", status_code=204)
def remove_member(project_id: str, membership_id: str, user: CurrentUser = Depends(require_user)) -> None:
    assert_role(user, project_id, Role.ADMIN)
    with session_scope() as s:
        m = s.get(Membership, membership_id)
        if not m or m.project_id != project_id:
            raise HTTPException(404)
        s.delete(m)
