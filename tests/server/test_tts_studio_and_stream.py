from __future__ import annotations

import json
import csv
from pathlib import Path
import numpy as np
import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from oas_core.manifest import TTSSample, AudioRef, LicenseInfo, Split
from oas_server.jobs.tts_finetune import _prepare_piper_dataset
from oas_server.serving.cache import clear as clear_cache


# ===========================================================================
# 1. Test Custom Phonetic Lexicon Expansion
# ===========================================================================

def test_lexicon_preprocessing_expansion(tmp_path: Path) -> None:
    # 1. Create a dummy sine wave audio file
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    import soundfile as sf
    audio_path = tmp_path / "dummy.wav"
    sf.write(str(audio_path), sine, sr)

    # 2. Build TTSSample with abbreviations
    sample = TTSSample(
        id="tts-1",
        license=LicenseInfo(spdx="CC0-1.0"),
        audio=AudioRef(
            uri=f"file://{audio_path}",
            sample_rate=sr,
            channels=1,
            duration_s=0.5
        ),
        text="Welcome to OAS on FastAPI today.",
        speaker_id="speaker-1",
        split=Split.TRAIN,
    )

    out_dir = tmp_path / "piper_dataset"
    out_dir.mkdir()

    # 3. Custom Lexicon configuration
    lexicon = {
        "OAS": "Oh-Ay-Es",
        "FastAPI": "Fast-Ay-Pee-Eye"
    }

    # 4. Prepare dataset
    class MockCtx:
        def log(self, msg): pass
        cancelled = False

    meta_path = _prepare_piper_dataset([sample], out_dir, sr, MockCtx(), lexicon=lexicon)

    # 5. Read prepared metadata and verify replacements occurred
    assert meta_path.exists()
    with meta_path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter="|")
        rows = list(reader)
    
    assert len(rows) == 1
    assert rows[0][0] == "tts-1"
    # Should replace OAS -> Oh-Ay-Es and FastAPI -> Fast-Ay-Pee-Eye
    assert rows[0][1] == "Welcome to Oh-Ay-Es on Fast-Ay-Pee-Eye today."
    assert rows[0][2] == "speaker-1"


# ===========================================================================
# 2. Test TTS Completions Sentence Chunk Streaming
# ===========================================================================

class MockTTSServer:
    def synth(self, text: str) -> tuple[np.ndarray, int]:
        # Yield silent sine wave chunk
        return np.zeros(22050, dtype="float32"), 22050


def test_tts_stream_endpoint(monkeypatch) -> None:
    clear_cache()

    # Monkeypatch load_tts to return MockTTSServer
    def mock_load(version_id: str) -> MockTTSServer:
        return MockTTSServer()
    
    monkeypatch.setattr("oas_server.routers.serve.load_tts", mock_load)

    from oas_server.routers.serve import tts_stream, SynthIn

    body = SynthIn(text="Hello world! This is a test. Audio stream now.")
    res = tts_stream("mock-tts-id", body)
    assert isinstance(res, StreamingResponse)

    import asyncio

    async def consume(gen):
        out = []
        async for chunk in gen:
            out.append(chunk)
        return out

    chunks = asyncio.run(consume(res.body_iterator))
    assert len(chunks) > 0

    # Parse and verify chunks
    data_elements = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                payload = json.loads(line.removeprefix("data: "))
                assert "audio" in payload
                assert "text" in payload
                data_elements.append(payload["text"])
    
    # Text should be split into sentences properly
    assert "Hello world!" in data_elements
    assert "This is a test." in data_elements
    assert "Audio stream now." in data_elements


# ===========================================================================
# 3. Test MOS Star Ratings database persistence
# ===========================================================================

def _client() -> TestClient:
    from oas_server.main import create_app
    return TestClient(create_app())


def test_record_mos_star_rating_endpoint() -> None:
    with _client() as c:
        # Create user, project, tts model and version
        r = c.post(
            "/auth/signup",
            json={"email": "tts_tester@oas.com", "password": "password123", "name": "TTS Tester"},
        )
        assert r.status_code == 201
        token = r.json()["access_token"]
        h = {"authorization": f"Bearer {token}"}

        proj = c.post("/projects", json={"slug": "tts-proj", "name": "TTS Project"}, headers=h).json()
        
        model = c.post(
            "/models",
            json={
                "project_id": proj["id"],
                "slug": "voice-model",
                "name": "Voice Model",
                "modality": "tts",
            },
            headers=h,
        ).json()

        mv = c.post(
            f"/models/{model['id']}/versions",
            json={
                "version": "1.0.0",
                "artifact_uri": "file:///mock/voice",
                "format": "piper-onnx",
            },
            headers=h,
        ).json()

        version_id = mv["id"]

        # A. Submit first rating: 5.0 stars
        r1 = c.post(f"/models/versions/{version_id}/mos", json={"score": 5.0}, headers=h)
        assert r1.status_code == 200, r1.text
        data1 = r1.json()
        assert data1["metrics"]["mos"] == 5.0
        assert data1["metrics"]["mos_ratings"] == [5.0]

        # B. Submit second rating: 3.0 stars
        r2 = c.post(f"/models/versions/{version_id}/mos", json={"score": 3.0}, headers=h)
        assert r2.status_code == 200, r2.text
        data2 = r2.json()
        # Average of 5.0 and 3.0 is 4.0
        assert data2["metrics"]["mos"] == 4.0
        assert data2["metrics"]["mos_ratings"] == [5.0, 3.0]
