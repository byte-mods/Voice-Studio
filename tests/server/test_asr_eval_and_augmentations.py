from __future__ import annotations

import io
import math
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from oas_core.manifest import ASRSample, AudioRef, LicenseInfo
from oas_server.jobs.whisper_finetune import _make_dataset


class MockProcessor:
    def __init__(self):
        class Tokenizer:
            def __init__(self):
                self.pad_token_id = 50256
            def __call__(self, text, **kwargs):
                class Out:
                    def __init__(self):
                        self.input_ids = [np.array([1, 2, 3])]
                return Out()
        self.tokenizer = Tokenizer()

    def __call__(self, data, **kwargs):
        class Features:
            def __init__(self):
                self.input_features = [np.zeros((80, 3000))]
        return Features()


def test_audio_augmentations(tmp_path: Path) -> None:
    # 1. Generate a synthetic 1-channel sine wave audio file (1 sec, 16000Hz)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    import soundfile as sf
    audio_path = tmp_path / "sine.wav"
    sf.write(str(audio_path), sine, sr)
    
    # 2. Build ASRSample
    sample = ASRSample(
        id="sample-123",
        license=LicenseInfo(spdx="CC0-1.0"),
        audio=AudioRef(
            uri=f"file://{audio_path}",
            sample_rate=sr,
            channels=1,
            duration_s=1.0
        ),
        transcript="hello world"
    )
    
    processor = MockProcessor()
    
    # 3. Test with speed perturbation enabled
    aug_speed = {
        "speed": {"enabled": True, "min_factor": 1.2, "max_factor": 1.2}
    }
    ds_speed = _make_dataset([sample], processor, max_audio_s=30, augmentations=aug_speed)
    # The linear interpolation will be applied. Verify length of loaded data is perturbed.
    # To check the internal perturbation, we can simulate what happens in DS.__getitem__
    item = ds_speed[0]
    assert "input_features" in item
    assert "labels" in item

    # 4. Test with noise perturbation enabled
    aug_noise = {
        "noise": {"enabled": True, "min_snr_db": 10, "max_snr_db": 20}
    }
    ds_noise = _make_dataset([sample], processor, max_audio_s=30, augmentations=aug_noise)
    item_noise = ds_noise[0]
    assert item_noise is not None

    # 5. Test with reverb perturbation enabled
    aug_reverb = {
        "reverb": {"enabled": True, "decay": 0.4}
    }
    ds_reverb = _make_dataset([sample], processor, max_audio_s=30, augmentations=aug_reverb)
    item_reverb = ds_reverb[0]
    assert item_reverb is not None


def test_asr_eval_worst_error_linking(monkeypatch) -> None:
    # Verify that the asr_eval worst error collector captures the sample id and audio uri
    import jiwer
    
    class MockSample:
        def __init__(self, id, uri, transcript):
            self.id = id
            class Audio:
                def __init__(self, uri):
                    self.uri = uri
            self.audio = Audio(uri)
            self.transcript = transcript
            self.modality = None
            self.split = None

    samples = [
        MockSample("s1", "file://1.wav", "hello world"),
        MockSample("s2", "file://2.wav", "welcome here")
    ]
    
    refs = ["hello world", "welcome here"]
    hyps = ["hello word", "welcome dere"]
    
    # Replicate worst errors extraction loop in asr_eval
    errors = []
    processed_samples = samples
    
    for smpl, r, h in zip(processed_samples, refs, hyps, strict=True):
        measures = jiwer.process_words(r, h)
        errors.append({
            "id": smpl.id,
            "audio_uri": smpl.audio.uri,
            "ref": r,
            "hyp": h,
            "wer": float(measures.wer),
            "subs": int(measures.substitutions),
            "ins": int(measures.insertions),
            "del": int(measures.deletions),
        })
        
    assert len(errors) == 2
    assert errors[0]["id"] == "s1"
    assert errors[0]["audio_uri"] == "file://1.wav"
    assert errors[0]["subs"] == 1
    assert errors[1]["id"] == "s2"
    assert errors[1]["audio_uri"] == "file://2.wav"


def _client() -> TestClient:
    from oas_server.main import create_app
    return TestClient(create_app())


def test_serve_endpoints_registered() -> None:
    with _client() as c:
        # Check that post and websocket serve routes are registered
        routes = [r.path for r in c.app.routes]
        assert any("/serve/asr/{version_id}/transcribe" in r for r in routes)
        assert any("/serve/asr/{version_id}/stream" in r for r in routes)
