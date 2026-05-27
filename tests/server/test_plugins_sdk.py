from __future__ import annotations

import os
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


@pytest.fixture(autouse=True)
def require_auth():
    # Force authentication checks
    os.environ["OAS_AUTH_REQUIRED"] = "true"
    yield
    os.environ["OAS_AUTH_REQUIRED"] = "false"


def test_list_plugins_and_scaffold_sdk() -> None:
    with _client() as c:
        # Sign up superuser (first user)
        su_signup = c.post("/auth/signup", json={"email": "su@team.com", "password": "password123"}).json()
        su_token = su_signup["access_token"]

        # Sign up regular user (non-superuser)
        reg_signup = c.post("/auth/signup", json={"email": "regular@team.com", "password": "password123"}).json()
        reg_token = reg_signup["access_token"]

        # 1. Test GET /plugins (should succeed for both)
        res = c.get("/plugins", headers={"authorization": f"Bearer {su_token}"})
        assert res.status_code == 200
        assert isinstance(res.json(), list)

        res = c.get("/plugins", headers={"authorization": f"Bearer {reg_token}"})
        assert res.status_code == 200

        # 2. Test POST /plugins/install security limits
        # Regular user attempts to install (should fail - 403)
        res = c.post(
            "/plugins/install",
            json={"source": "dummy-pypi-package"},
            headers={"authorization": f"Bearer {reg_token}"},
        )
        assert res.status_code == 403

        # 3. Test POST /plugins/scaffold
        # Regular user attempts to scaffold (should fail - 403)
        res = c.post(
            "/plugins/scaffold",
            json={
                "name": "Scaffold Custom task",
                "kind": "custom_scaffold",
                "description": "Boilerplate custom tasks desc"
            },
            headers={"authorization": f"Bearer {reg_token}"},
        )
        assert res.status_code == 403

        # Superuser scaffolds a plugin (should succeed - 200)
        res = c.post(
            "/plugins/scaffold",
            json={
                "name": "Scaffold Custom task",
                "kind": "custom_scaffold",
                "description": "Boilerplate custom tasks desc"
            },
            headers={"authorization": f"Bearer {su_token}"},
        )
        assert res.status_code == 200, res.text
        scaffold_out = res.json()
        assert scaffold_out["success"] is True
        assert scaffold_out["slug"] == "scaffold-custom-task"
        dest_path = Path(scaffold_out["destination"])
        assert dest_path.exists()

        # Assert on-disk files structure
        assert (dest_path / "pyproject.toml").exists()
        assert (dest_path / "oas_scaffold_custom_task" / "__init__.py").exists()
        assert (dest_path / "oas_scaffold_custom_task" / "handlers.py").exists()

        handlers_code = (dest_path / "oas_scaffold_custom_task" / "handlers.py").read_text()
        assert "def custom_scaffold_handler" in handlers_code

        # Cleanup scaffolded folder
        if dest_path.exists():
            shutil.rmtree(dest_path)
