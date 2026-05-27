"""Server test config — reuse the core isolation fixture."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OAS_DB_URL", f"sqlite:///{tmp_path / 'oas.db'}")
    monkeypatch.setenv("OAS_WORKER_CONCURRENCY", "2")

    from oas_core import db
    from oas_core import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.SessionLocal.configure(bind=None)
