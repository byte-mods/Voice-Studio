from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from oas.client import Studio


@pytest.fixture()
def studio() -> Studio:
    from oas_core.db import init_db
    from oas_server.main import create_app

    init_db()
    client = TestClient(create_app())
    s = Studio(client=client)
    yield s
    client.close()


def test_end_to_end_flow(studio: Studio) -> None:
    p = studio.projects.create(slug="sdk-demo", name="SDK Demo")
    d = studio.datasets.create(p.id, "asr1", "ASR 1", "asr")
    v = studio.datasets.add_version(d.id, "0.1.0", "file:///tmp/m", num_samples=3)
    assert v.num_samples == 3

    m = studio.models.create(p.id, "whisper", "Whisper", "asr")
    mv = studio.models.publish(m.id, "0.1.0", "file:///tmp/x", metrics={"wer": 0.2})
    assert mv.stage == "dev"
    mv2 = studio.models.set_stage(mv.id, "staging")
    assert mv2.stage == "staging"

    info = studio.system_info()
    assert "noop" in info["handlers"]
