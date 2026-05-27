from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from oas_server.jobs.codec_train import codec_train_handler
from oas_server.jobs.s2s_native_finetune import s2s_native_finetune_handler
from oas_core.queue.backend import JobContext


def _client() -> TestClient:
    from oas_server.main import create_app
    return TestClient(create_app())


def test_codec_training_loop_simulation(tmp_path: Path) -> None:
    # Setup test database and dataset versions
    with _client() as c:
        pid = c.post("/projects", json={"slug": "native-omni", "name": "Native S2S Project"}).json()["id"]
        
        # Create dataset
        did = c.post("/datasets", json={"project_id": pid, "slug": "speech-wavs", "name": "Speech Wavs", "modality": "s2s"}).json()["id"]
        
        # Create version
        vid = c.post(f"/datasets/{did}/versions", json={"version": "1.0.0", "description": "codec training", "manifest_uri": "file://" + str(tmp_path)}).json()["id"]

        logs_captured: list[str] = []
        ctx = JobContext(
            job_id="job-codec-123",
            run_id="run-codec-123",
            kind="codec_train",
            config={
                "dataset_version_id": vid,
                "base_model": "facebook/encodec_24khz",
                "hyperparameters": {
                    "epochs": 2,
                    "num_quantizers": 4,
                    "codebook_size": 512,
                    "sample_rate": 16000
                }
            },
            artifacts_dir=str(tmp_path / "artifacts"),
            logs_dir=str(tmp_path / "logs"),
        )
        ctx.add_log_sink(lambda m: logs_captured.append(m))
        
        # Write mock manifest files satisfying Pydantic ManifestHeader
        (tmp_path / "manifest.json").write_text('{"modality": "s2s", "dataset_id": "speech-wavs", "name": "Speech Wavs"}')
        (tmp_path / "samples.jsonl").touch()

        res = codec_train_handler(ctx)
        assert res["epochs"] == 2
        assert res["sample_rate"] == 16000
        assert "loss" in res
        assert "spectral_reconstruct_loss" in res
        assert len(logs_captured) > 0
        assert any("quantizer" in l.lower() for l in logs_captured)

        # Verify custom config file was created
        config_path = tmp_path / "artifacts" / "final" / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            cfg_data = json.load(f)
            assert cfg_data["base_model"] == "facebook/encodec_24khz"
            assert cfg_data["num_quantizers"] == 4
            assert cfg_data["codebook_size"] == 512


def test_duplex_and_voice_preservation_parsing(tmp_path: Path, monkeypatch) -> None:
    # Verify that the native SFT job loop compiles duplex toggles cleanly
    mock_dataset_root = tmp_path / "dataset"
    mock_dataset_root.mkdir()
    
    with _client() as c:
        pid = c.post("/projects", json={"slug": "sft-duplex", "name": "SFT Duplex"}).json()["id"]
        did = c.post("/datasets", json={"project_id": pid, "slug": "dialogue-turns", "name": "Dialogue Turns", "modality": "s2s"}).json()["id"]
        vid = c.post(f"/datasets/{did}/versions", json={"version": "1.0.0", "description": "dialogue", "manifest_uri": "file://" + str(mock_dataset_root)}).json()["id"]

        # Run native SFT handler with mock imports
        monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *args, **kwargs: type("Proc", (), {"apply_chat_template": lambda *a, **k: {"input_ids": "fake"}}))
        monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *args, **kwargs: type("Model", (), {"gradient_checkpointing_enable": lambda *a, **k: None}))
        monkeypatch.setattr("peft.get_peft_model", lambda model, *args, **kwargs: type("Peft", (), {"print_trainable_parameters": lambda *a: None, "train": lambda: None, "save_model": lambda *a: None}))
        
        logs_captured: list[str] = []
        ctx = JobContext(
            job_id="job-sft-456",
            run_id="run-sft-456",
            kind="s2s_native_finetune",
            config={
                "dataset_version_id": vid,
                "base_model": "Qwen/Qwen2.5-Omni-3B",
                "duplex_mode": True,
                "preserve_voice": True,
                "training": {
                    "epochs": 1,
                    "batch_size": 1
                }
            },
            artifacts_dir=str(tmp_path / "sft_artifacts"),
            logs_dir=str(tmp_path / "sft_logs"),
        )
        ctx.add_log_sink(lambda m: logs_captured.append(m))
        
        # Write mock manifest files satisfying Pydantic ManifestHeader
        (mock_dataset_root / "manifest.json").write_text('{"modality": "s2s", "dataset_id": "dialogue-turns", "name": "Dialogue Turns"}')
        (mock_dataset_root / "samples.jsonl").touch()
        
        # We catch ValueError("no S2S train samples in manifest") which indicates
        # the handler parsed configs, loaded DB, parsed duplex/voice, and hit
        # S2S dataset checks cleanly without throwing compilation or import errors!
        with pytest.raises(ValueError, match="no S2S train samples in manifest"):
            s2s_native_finetune_handler(ctx)

        assert any("Enforcing speaking/listening overlapping attention masks" in l for l in logs_captured)
        assert any("Injecting reference speaker vocal mappings" in l for l in logs_captured)
