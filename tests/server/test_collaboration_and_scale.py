from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient
from oas_core.db import Role


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


@pytest.fixture(autouse=True)
def require_auth():
    # Force authentication checks for all RBAC tests
    os.environ["OAS_AUTH_REQUIRED"] = "true"
    yield
    os.environ["OAS_AUTH_REQUIRED"] = "false"


def test_rbac_access_control_and_credential_redaction() -> None:
    with _client() as c:
        # 1. Sign up Admin (first ever user → superuser bypass inside signup is set to True,
        # but let's test project-level RBAC by using regular membership roles).
        admin_signup = c.post("/auth/signup", json={"email": "admin@team.com", "password": "password123"}).json()
        admin_token = admin_signup["access_token"]
        admin_id = admin_signup["user"]["id"]

        # Create Editor & Viewer users
        editor_signup = c.post("/auth/signup", json={"email": "editor@team.com", "password": "password123"}).json()
        editor_token = editor_signup["access_token"]
        editor_id = editor_signup["user"]["id"]

        viewer_signup = c.post("/auth/signup", json={"email": "viewer@team.com", "password": "password123"}).json()
        viewer_token = viewer_signup["access_token"]
        viewer_id = viewer_signup["user"]["id"]

        non_member_signup = c.post("/auth/signup", json={"email": "external@team.com", "password": "password123"}).json()
        non_member_token = non_member_signup["access_token"]

        # Admin creates project (becomes ADMIN membership)
        proj = c.post(
            "/projects",
            json={"slug": "collab-scale", "name": "Collaboration and Cloud Scale"},
            headers={"authorization": f"Bearer {admin_token}"},
        ).json()
        proj_id = proj["id"]

        # Add Editor and Viewer to the project
        c.post(
            f"/auth/projects/{proj_id}/members",
            json={"user_id": editor_id, "role": "editor"},
            headers={"authorization": f"Bearer {admin_token}"},
        )
        c.post(
            f"/auth/projects/{proj_id}/members",
            json={"user_id": viewer_id, "role": "viewer"},
            headers={"authorization": f"Bearer {admin_token}"},
        )

        # 2. Assert role access limits
        # EDITOR attempts to add member (should fail - 403)
        res = c.post(
            f"/auth/projects/{proj_id}/members",
            json={"user_id": viewer_id, "role": "admin"},
            headers={"authorization": f"Bearer {editor_token}"},
        )
        assert res.status_code == 403

        # VIEWER attempts to add member (should fail - 403)
        res = c.post(
            f"/auth/projects/{proj_id}/members",
            json={"user_id": editor_id, "role": "editor"},
            headers={"authorization": f"Bearer {viewer_token}"},
        )
        assert res.status_code == 403

        # NON-MEMBER attempts to view members (should fail - 403)
        res = c.get(
            f"/auth/projects/{proj_id}/members",
            headers={"authorization": f"Bearer {non_member_token}"},
        )
        assert res.status_code == 403

        # VIEWER attempts to view members (should succeed - 200)
        res = c.get(
            f"/auth/projects/{proj_id}/members",
            headers={"authorization": f"Bearer {viewer_token}"},
        )
        assert res.status_code == 200
        members_list = res.json()
        assert len(members_list) >= 3

        # 3. Assert project settings update and credential redaction
        creds = {
            "cloud_providers": {
                "modal": {
                    "token_id": "ak-modal-client-id",
                    "token_secret": "super-secret-modal-key-12345",
                },
                "runpod": {
                    "api_key": "runpod-secret-api-key-abcdef",
                }
            }
        }

        # EDITOR attempts to update project settings (should fail - 403)
        res = c.put(
            f"/projects/{proj_id}",
            json={"settings": creds},
            headers={"authorization": f"Bearer {editor_token}"},
        )
        assert res.status_code == 403

        # ADMIN updates project settings with credentials (should succeed - 200)
        res = c.put(
            f"/projects/{proj_id}",
            json={"settings": creds},
            headers={"authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        saved_proj = res.json()

        # Assert that settings returned in payload are REDACTED
        saved_providers = saved_proj["settings"]["cloud_providers"]
        assert saved_providers["modal"]["token_id"] == "ak-modal-client-id"
        assert saved_providers["modal"]["token_secret"] == "********"
        assert saved_providers["runpod"]["api_key"] == "********"

        # Try to update with "********" placeholder, check that the original value is preserved
        res = c.put(
            f"/projects/{proj_id}",
            json={
                "settings": {
                    "cloud_providers": {
                        "modal": {
                            "token_id": "ak-modal-client-id-updated",
                            "token_secret": "********",
                        },
                        "runpod": {
                            "api_key": "********",
                        }
                    }
                }
            },
            headers={"authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        updated_proj = res.json()
        assert updated_proj["settings"]["cloud_providers"]["modal"]["token_id"] == "ak-modal-client-id-updated"
        assert updated_proj["settings"]["cloud_providers"]["modal"]["token_secret"] == "********"


def test_compute_endpoints_and_telemetry() -> None:
    with _client() as c:
        admin_signup = c.post("/auth/signup", json={"email": "owner@team.com", "password": "password123"}).json()
        admin_token = admin_signup["access_token"]

        proj = c.post(
            "/projects",
            json={"slug": "compute-test", "name": "Compute and Pricing Dials"},
            headers={"authorization": f"Bearer {admin_token}"},
        ).json()
        proj_id = proj["id"]

        # Register credentials first so test does not return warning
        c.put(
            f"/projects/{proj_id}",
            json={
                "settings": {
                    "cloud_providers": {
                        "modal": {"token_id": "abc", "token_secret": "xyz"},
                        "runpod": {"api_key": "xyz"}
                    }
                }
            },
            headers={"authorization": f"Bearer {admin_token}"},
        )

        # 1. Test liveness check endpoint
        res = c.post(
            f"/projects/{proj_id}/compute/test",
            json={"provider": "modal"},
            headers={"authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        test_out = res.json()
        assert test_out["status"] == "online"
        assert test_out["latency_ms"] > 0
        assert len(test_out["gpus"]) > 0

        # Test missing config provider
        res = c.post(
            f"/projects/{proj_id}/compute/test",
            json={"provider": "slurm"},
            headers={"authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "offline"

        # 2. Test compute telemetry dials endpoint
        res = c.get(
            f"/projects/{proj_id}/compute/telemetry",
            headers={"authorization": f"Bearer {admin_token}"},
        )
        assert res.status_code == 200
        telemetry = res.json()
        assert telemetry["total_gpu_hours"] > 0
        assert telemetry["total_cost_usd"] > 0
        assert len(telemetry["runs_history"]) > 0
        assert len(telemetry["active_nodes"]) > 0


def test_scoped_audit_logs() -> None:
    with _client() as c:
        admin_signup = c.post("/auth/signup", json={"email": "boss@team.com", "password": "password123"}).json()
        admin_token = admin_signup["access_token"]

        viewer_signup = c.post("/auth/signup", json={"email": "auditor@team.com", "password": "password123"}).json()
        viewer_token = viewer_signup["access_token"]
        viewer_id = viewer_signup["user"]["id"]

        external_signup = c.post("/auth/signup", json={"email": "hacker@team.com", "password": "password123"}).json()
        external_token = external_signup["access_token"]

        # Create project
        proj = c.post(
            "/projects",
            json={"slug": "audited-proj", "name": "Audit Scopes"},
            headers={"authorization": f"Bearer {admin_token}"},
        ).json()
        proj_id = proj["id"]

        # Add viewer to project
        c.post(
            f"/auth/projects/{proj_id}/members",
            json={"user_id": viewer_id, "role": "viewer"},
            headers={"authorization": f"Bearer {admin_token}"},
        )

        # 1. Non-superuser calls global audit endpoint without project_id (should fail - 403)
        res = c.get(
            "/audit",
            headers={"authorization": f"Bearer {viewer_token}"},
        )
        assert res.status_code == 403

        # 2. Project member (viewer) calls audit with project_id filter (should succeed - 200)
        res = c.get(
            f"/audit?project_id={proj_id}",
            headers={"authorization": f"Bearer {viewer_token}"},
        )
        assert res.status_code == 200
        assert isinstance(res.json(), list)

        # 3. External user calls audit with project_id filter (should fail - 403)
        res = c.get(
            f"/audit?project_id={proj_id}",
            headers={"authorization": f"Bearer {external_token}"},
        )
        assert res.status_code == 403
