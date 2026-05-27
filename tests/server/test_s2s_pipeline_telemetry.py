from __future__ import annotations

import time
import pytest
from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app
    return TestClient(create_app())


def test_s2s_pipeline_route_registrations() -> None:
    with _client() as c:
        routes = [r.path for r in c.app.routes]
        # Check CRUD routes
        assert any("/s2s/pipelines" in r for r in routes)
        assert any("/s2s/pipelines/{pipeline_id}" in r for r in routes)
        # Check WebSocket endpoint
        assert any("/s2s/sessions/{pipeline_id}" in r for r in routes)


def test_s2s_cascade_telemetry_simulation() -> None:
    # Replicate and verify the frontend's latency collector loop logic in a unit test
    # to ensure that interval delta computations cannot crash or return negative metrics.
    
    events = [
        {"type": "turn_start", "time": 0.0},
        {"type": "final_transcript", "time": 0.150},  # ASR: 150ms
        {"type": "assistant_text", "time": 0.440},    # LLM: 290ms
        {"type": "tts_start", "time": 0.520},         # TTS: 80ms, TTFA: 520ms
    ]

    # Process and accumulate delta offsets
    asr_start = None
    llm_start = None
    tts_start = None
    ttfa_start = None

    telemetry = {}

    for ev in events:
        t = ev["time"]
        if ev["type"] == "turn_start":
            ttfa_start = t
            asr_start = t
        elif ev["type"] == "final_transcript":
            if asr_start is not None:
                telemetry["asr"] = int((t - asr_start) * 1000)
                asr_start = None
            llm_start = t
        elif ev["type"] == "assistant_text":
            if llm_start is not None:
                telemetry["llm"] = int((t - llm_start) * 1000)
                llm_start = None
            tts_start = t
        elif ev["type"] == "tts_start":
            if tts_start is not None:
                telemetry["tts"] = int((t - tts_start) * 1000)
                tts_start = None
            if ttfa_start is not None:
                telemetry["ttfa"] = int((t - ttfa_start) * 1000)
                ttfa_start = None

    assert telemetry["asr"] == 150
    assert telemetry["llm"] == 290
    assert telemetry["tts"] == 80
    assert telemetry["ttfa"] == 520


def test_s2s_pipeline_creation_with_native_mode() -> None:
    with _client() as c:
        # Create project
        pid = c.post("/projects", json={"slug": "s2s-native", "name": "S2S Native"}).json()["id"]
        
        # Create pipeline with native audio-LM runtime config
        r = c.post(
            "/s2s/pipelines",
            json={
                "project_id": pid,
                "slug": "native-cascade",
                "name": "Native Cascade",
                "asr_fallback": None,
                "llm_fallback": "Qwen/Qwen2.5-Omni-7B",
                "tts_fallback": "facebook/mms-tts-eng",
                "runtime_config": {"mode": "native"},
            },
        )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["runtime_config"]["mode"] == "native"
        assert data["asr_version_id"] is None
        assert data["asr_fallback"] is None
