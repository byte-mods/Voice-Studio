from __future__ import annotations

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


def test_new_handlers_and_routes_registered() -> None:
    with _client() as c:
        kinds = c.get("/jobs/handlers").json()
        assert "s2s_native_finetune" in kinds

        # Confirm new routes are mounted by checking OpenAPI.
        spec = c.get("/openapi.json").json()
        paths = set(spec["paths"].keys())
        assert "/serve/tts/{version_id}/synthesize" in paths
        assert "/serve/asr/{version_id}/transcribe" in paths
        assert "/serve/llm/{version_id}/v1/chat/completions" in paths
        assert "/plans/finetune" in paths

        r = c.post("/plans/finetune", json={})
        assert r.status_code in (400, 404, 422)


def test_rbac_blocks_unauthorized_dataset_create(monkeypatch) -> None:
    monkeypatch.setenv("OAS_AUTH_REQUIRED", "true")
    with _client() as c:
        r = c.post(
            "/datasets",
            json={"project_id": "fake", "slug": "x", "name": "x", "modality": "asr"},
        )
        assert r.status_code == 401, r.text


def test_creator_becomes_admin_and_can_add_members() -> None:
    with _client() as c:
        admin = c.post("/auth/signup", json={"email": "creator@x.com", "password": "password123"}).json()
        admin_token = admin["access_token"]
        proj = c.post(
            "/projects",
            json={"slug": "owned", "name": "Owned"},
            headers={"authorization": f"Bearer {admin_token}"},
        ).json()

        second = c.post("/auth/signup", json={"email": "x@y.com", "password": "password123"}).json()["user"]
        r = c.post(
            f"/auth/projects/{proj['id']}/members",
            json={"user_id": second["id"], "role": "editor"},
            headers={"authorization": f"Bearer {admin_token}"},
        )
        assert r.status_code == 201, r.text
