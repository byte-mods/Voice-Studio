from __future__ import annotations

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


def test_signup_login_me_and_first_is_superuser() -> None:
    with _client() as c:
        # Signup
        r = c.post("/auth/signup", json={"email": "a@b.com", "password": "password123"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["user"]["is_superuser"] is True  # first signup → superuser
        token = body["access_token"]

        # /me with bearer
        r = c.get("/auth/me", headers={"authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["email"] == "a@b.com"

        # Login again with same credentials
        r = c.post("/auth/login", json={"email": "a@b.com", "password": "password123"})
        assert r.status_code == 200
        assert r.json()["user"]["email"] == "a@b.com"


def test_login_rejects_bad_password() -> None:
    with _client() as c:
        c.post("/auth/signup", json={"email": "x@y.com", "password": "password123"})
        r = c.post("/auth/login", json={"email": "x@y.com", "password": "wrong-password"})
        assert r.status_code == 401


def test_second_user_is_not_superuser_and_can_become_member() -> None:
    with _client() as c:
        c.post("/auth/signup", json={"email": "admin@x.com", "password": "password123"})
        admin_token = c.post(
            "/auth/login", json={"email": "admin@x.com", "password": "password123"}
        ).json()["access_token"]

        second = c.post("/auth/signup", json={"email": "user@x.com", "password": "password123"}).json()
        assert second["user"]["is_superuser"] is False
        second_id = second["user"]["id"]

        # Admin creates a project
        proj = c.post(
            "/projects",
            json={"slug": "rbac", "name": "RBAC"},
            headers={"authorization": f"Bearer {admin_token}"},
        ).json()

        # Add second user as editor
        r = c.post(
            f"/auth/projects/{proj['id']}/members",
            json={"user_id": second_id, "role": "editor"},
            headers={"authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 201, r.text

        listed = c.get(
            f"/auth/projects/{proj['id']}/members",
            headers={"authorization": f"Bearer {admin_token}"},
        ).json()
        assert any(m["user_id"] == second_id for m in listed)


def test_anonymous_works_when_auth_not_required() -> None:
    # By default OAS_AUTH_REQUIRED is unset → anonymous superuser shortcut.
    with _client() as c:
        r = c.get("/auth/me")
        assert r.status_code == 200
        assert r.json()["email"] == "anonymous@local"
