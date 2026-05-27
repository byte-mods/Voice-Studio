from __future__ import annotations

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


def test_response_has_request_id_header() -> None:
    with _client() as c:
        r = c.get("/healthz")
        rid = r.headers.get("x-request-id")
        assert rid, "expected x-request-id on responses"
        assert len(rid) >= 8


def test_inbound_request_id_is_honored() -> None:
    with _client() as c:
        r = c.get("/healthz", headers={"x-request-id": "my-custom-id-1234"})
        assert r.headers.get("x-request-id") == "my-custom-id-1234"


def test_audit_log_records_request_id() -> None:
    with _client() as c:
        c.post("/auth/signup", json={"email": "rid@x.com", "password": "password123"})
        token = c.post(
            "/auth/login", json={"email": "rid@x.com", "password": "password123"}
        ).json()["access_token"]
        h = {"authorization": f"Bearer {token}", "x-request-id": "rid-test-abc"}
        c.post("/projects", json={"slug": "rid-p", "name": "RidP"}, headers=h)

        logs = c.get("/audit", headers=h).json()
        # At least one log entry should carry our custom request id.
        assert any(e.get("request_id") == "rid-test-abc" for e in logs)
