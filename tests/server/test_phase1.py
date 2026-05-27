"""Phase 1 endpoints: uploads, samples preview, hf_import registration."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from oas_core.manifest import (
    ASRSample,
    AudioRef,
    LicenseInfo,
    ManifestHeader,
    ManifestWriter,
    Modality,
)


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


def test_hf_import_handler_is_registered() -> None:
    with _client() as c:
        kinds = c.get("/jobs/handlers").json()
        assert "hf_import" in kinds
        assert "asr_bootstrap" in kinds
        assert "whisper_finetune" in kinds
        assert "tts_finetune_piper" in kinds
        assert "llm_finetune_sft" in kinds
        assert "noop" in kinds


def test_upload_endpoint() -> None:
    with _client() as c:
        files = {"file": ("hello.txt", io.BytesIO(b"hi there"), "text/plain")}
        r = c.post("/uploads", files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["filename"] == "hello.txt"
        assert body["size"] == 8
        assert body["sha256"]
        assert body["uri"].startswith("file://")


def test_samples_preview_pagination(tmp_path: Path) -> None:
    # Build a manifest on disk, register it as a DatasetVersion, then page samples via the API.
    from oas_core.db import Dataset, DatasetVersion, Project, init_db, session_scope
    from oas_core.settings import get_settings

    init_db()
    settings = get_settings()
    with session_scope() as s:
        p = Project(slug="phase1", name="Phase1")
        s.add(p)
        s.flush()
        d = Dataset(project_id=p.id, slug="asr", name="ASR", modality=Modality.ASR)
        s.add(d)
        s.flush()
        dataset_id = d.id

    manifest_root = settings.datasets_dir / dataset_id / "0.1.0"
    manifest_root.mkdir(parents=True, exist_ok=True)
    header = ManifestHeader(dataset_id=dataset_id, name="t", modality=Modality.ASR)
    with ManifestWriter(manifest_root, header) as w:
        for i in range(7):
            w.add(
                ASRSample(
                    license=LicenseInfo(spdx="CC0-1.0"),
                    audio=AudioRef(
                        uri=f"file:///fake/{i}.wav", sample_rate=16000, duration_s=1.0
                    ),
                    transcript=f"sample {i}",
                )
            )

    with session_scope() as s:
        v = DatasetVersion(
            dataset_id=dataset_id,
            version="0.1.0",
            manifest_uri=f"file://{manifest_root}",
            num_samples=7,
        )
        s.add(v)
        s.flush()
        version_id = v.id

    with _client() as c:
        r = c.get(f"/datasets/versions/{version_id}/samples?offset=2&limit=3")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 7
        assert body["offset"] == 2
        assert len(body["items"]) == 3
        assert body["items"][0]["transcript"] == "sample 2"

        # Patch the middle sample's transcript
        target_id = body["items"][0]["id"]
        r = c.patch(
            f"/datasets/versions/{version_id}/samples/{target_id}",
            json={"transcript": "FIXED"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["transcript"] == "FIXED"

        # Verify on next read
        page = c.get(f"/datasets/versions/{version_id}/samples?offset=2&limit=1").json()
        assert page["items"][0]["transcript"] == "FIXED"
