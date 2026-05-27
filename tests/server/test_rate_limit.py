from __future__ import annotations

from fastapi.testclient import TestClient


def test_rate_limit_kicks_in(monkeypatch) -> None:
    monkeypatch.setenv("OAS_SERVE_RATE_RPS", "0.01")  # ~one token every 100s
    monkeypatch.setenv("OAS_SERVE_RATE_BURST", "2")
    monkeypatch.setenv("OAS_SERVE_RATE_SCOPE", "/serve")

    from oas_server.main import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as c:
        # First two requests within burst → not 429 (may be 404/422/500 from missing model).
        for _ in range(2):
            r = c.post("/serve/tts/missing/synthesize", json={"text": "hi"})
            assert r.status_code != 429, r.text
        # Third should hit the limit.
        r = c.post("/serve/tts/missing/synthesize", json={"text": "hi"})
        assert r.status_code == 429, r.text
        assert "rate limit" in r.text.lower()
