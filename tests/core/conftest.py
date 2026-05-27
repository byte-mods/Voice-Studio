"""Test config: isolate every test's DB + data dir."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OAS_DB_URL", f"sqlite:///{tmp_path / 'oas.db'}")
    monkeypatch.setenv("OAS_WORKER_CONCURRENCY", "2")

    # Clear cached settings + engine so each test gets fresh ones.
    from oas_core import db
    from oas_core import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.SessionLocal.configure(bind=None)


@pytest.fixture()
def db_engine():
    from oas_core.db import init_db

    return init_db()


@pytest.fixture()
def project_id(db_engine):
    from oas_core.db import Project, session_scope

    with session_scope() as s:
        p = Project(slug="test", name="Test")
        s.add(p)
        s.flush()
        return p.id
