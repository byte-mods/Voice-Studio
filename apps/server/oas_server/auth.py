"""Authentication + RBAC for the studio.

Design choices:
- Email + password (PBKDF2-SHA256 via stdlib `hashlib`, no extra deps).
- JWT-style signed tokens using `itsdangerous`'s URLSafeTimedSerializer if
  installed, otherwise a self-contained HMAC-SHA256 token.
- Bearer token in the `Authorization` header (or `?access_token=` for
  WebSocket convenience).
- Per-project RBAC via `Membership.role` (viewer / editor / admin).
- A single optional superuser bypass for bootstrap.

When `OAS_AUTH_REQUIRED=false` (the default while the studio is still in
single-user mode) the dependencies short-circuit to an "anonymous" admin
identity so existing flows keep working.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from fastapi import Depends, Header, HTTPException, status
from oas_core.db import Membership, Role, User, session_scope
from oas_core.settings import get_settings
from sqlalchemy import select

_DEFAULT_TTL_S = 60 * 60 * 24 * 7  # 7 days


def _secret_key() -> bytes:
    s = os.environ.get("OAS_AUTH_SECRET")
    if not s:
        # Stable per-install secret derived from data_dir if not set.
        settings = get_settings()
        s = hashlib.sha256(f"oas-secret-{settings.data_dir}".encode()).hexdigest()
    return s.encode()


def _is_auth_required() -> bool:
    return os.environ.get("OAS_AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"pbkdf2_sha256$200000${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, digest_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(iters))
        return hmac.compare_digest(candidate, expected)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Token issuance / verification (compact JWS-ish, no external dep)
# ---------------------------------------------------------------------------


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64u_dec(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def issue_token(user_id: str, *, ttl_s: int = _DEFAULT_TTL_S) -> str:
    payload = {"sub": user_id, "iat": int(time.time()), "exp": int(time.time()) + ttl_s}
    body = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(_secret_key(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64u(sig)}"


def verify_token(token: str) -> dict[str, Any] | None:
    try:
        body, sig = token.split(".")
    except ValueError:
        return None
    expected = hmac.new(_secret_key(), body.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64u_dec(sig)):
        return None
    try:
        payload = json.loads(_b64u_dec(body))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return cast(dict[str, Any], payload)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


@dataclass
class CurrentUser:
    id: str
    email: str
    name: str | None
    is_superuser: bool
    anonymous: bool = False

    def is_admin(self) -> bool:
        return self.is_superuser


_ANON = CurrentUser(id="anonymous", email="anonymous@local", name="Anonymous", is_superuser=True, anonymous=True)


async def require_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    # Always honor a valid bearer token, even in anonymous mode — so an admin
    # who is already signed in keeps their identity for audit/RBAC purposes.
    if authorization and authorization.lower().startswith("bearer "):
        payload = verify_token(authorization.split(None, 1)[1].strip())
        if payload:
            with session_scope() as s:
                u = s.get(User, payload["sub"])
                if u:
                    return CurrentUser(
                        id=u.id, email=u.email, name=u.name, is_superuser=bool(u.is_superuser)
                    )
        if _is_auth_required():
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    if not _is_auth_required():
        return _ANON
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")


_ROLE_RANK = {Role.VIEWER: 0, Role.EDITOR: 1, Role.ADMIN: 2}


def _role_for(user: CurrentUser, project_id: str) -> Role | None:
    if user.anonymous or user.is_superuser:
        return Role.ADMIN
    with session_scope() as s:
        m = s.scalar(
            select(Membership).where(
                Membership.project_id == project_id, Membership.user_id == user.id
            )
        )
        return m.role if m else None


def assert_role(user: CurrentUser, project_id: str, min_role: Role) -> None:
    """Imperative check. Use inside endpoints where the project id comes from
    the request body rather than the path."""
    role = _role_for(user, project_id)
    if role is None or _ROLE_RANK[role] < _ROLE_RANK[min_role]:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"requires role >= {min_role.value} on project {project_id}",
        )


def require_project_role(min_role: Role) -> Callable[..., Any]:
    """Returns a FastAPI dependency that enforces the user's role on the
    project named in the path. The dependency expects the FastAPI route to
    receive `project_id: str` as a path parameter or query/body field — the
    caller passes it via kwargs.
    """

    async def dep(project_id: str, user: CurrentUser = Depends(require_user)) -> CurrentUser:
        role = _role_for(user, project_id)
        if role is None or _ROLE_RANK[role] < _ROLE_RANK[min_role]:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"requires role >= {min_role.value} on project {project_id}",
            )
        return user

    return dep
