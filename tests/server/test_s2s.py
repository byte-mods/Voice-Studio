from __future__ import annotations

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


def test_s2s_pipeline_crud() -> None:
    with _client() as c:
        pid = c.post("/projects", json={"slug": "s2s", "name": "S2S"}).json()["id"]
        r = c.post(
            "/s2s/pipelines",
            json={
                "project_id": pid,
                "slug": "default",
                "name": "Default",
                "asr_fallback": "openai/whisper-tiny",
                "llm_fallback": "Qwen/Qwen2.5-0.5B-Instruct",
                "tts_fallback": "facebook/mms-tts-eng",
                "system_prompt": "be brief",
                "vad_config": {},
                "runtime_config": {},
            },
        )
        assert r.status_code == 201, r.text
        pipeline = r.json()
        assert pipeline["slug"] == "default"

        listed = c.get("/s2s/pipelines", params={"project_id": pid}).json()
        assert len(listed) == 1

        got = c.get(f"/s2s/pipelines/{pipeline['id']}").json()
        assert got["name"] == "Default"


def test_s2s_pipeline_unique_slug() -> None:
    with _client() as c:
        pid = c.post("/projects", json={"slug": "s2s2", "name": "S2S2"}).json()["id"]
        body = {
            "project_id": pid,
            "slug": "dup",
            "name": "A",
            "asr_fallback": "openai/whisper-tiny",
            "llm_fallback": "Qwen/Qwen2.5-0.5B-Instruct",
            "tts_fallback": "facebook/mms-tts-eng",
        }
        assert c.post("/s2s/pipelines", json=body).status_code == 201
        assert c.post("/s2s/pipelines", json={**body, "name": "B"}).status_code == 409
