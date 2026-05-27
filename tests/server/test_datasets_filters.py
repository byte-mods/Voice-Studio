"""Tests for dataset quality filters.

Verifies filtering by audio duration, signal-to-noise ratio (SNR),
quality score, and transcription character length via the API.
"""

from __future__ import annotations

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


def _seed_version(samples: int = 5, durations: list[float] | None = None, snrs: list[float | None] | None = None, quality_scores: list[float | None] | None = None, transcripts: list[str] | None = None) -> tuple[str, str, Path]:
    """Create a Project, Dataset (ASR), and one DatasetVersion with customizable sample features."""
    from oas_core.db import Dataset, DatasetVersion, Project, init_db, session_scope
    from oas_core.settings import get_settings

    init_db()
    settings = get_settings()
    with session_scope() as s:
        p = Project(slug="filters-test", name="Filters Test")
        s.add(p)
        s.flush()
        d = Dataset(project_id=p.id, slug="asr", name="ASR", modality=Modality.ASR)
        s.add(d)
        s.flush()
        dataset_id = d.id

    manifest_root = settings.datasets_dir / dataset_id / "0.1.0"
    manifest_root.mkdir(parents=True, exist_ok=True)
    header = ManifestHeader(
        dataset_id=dataset_id,
        dataset_version="0.1.0",
        name="filters:0.1.0",
        modality=Modality.ASR,
        license_default=LicenseInfo(spdx="CC-BY-4.0"),
    )
    with ManifestWriter(manifest_root, header) as w:
        for i in range(samples):
            duration = durations[i] if durations else 2.0
            snr = snrs[i] if snrs else None
            quality = quality_scores[i] if quality_scores else None
            transcript = transcripts[i] if transcripts else f"sample {i}"
            w.add(
                ASRSample(
                    license=LicenseInfo(spdx="CC-BY-4.0"),
                    audio=AudioRef(
                        uri=f"file:///fake/{i}.wav", sample_rate=16000, duration_s=duration, snr_db=snr
                    ),
                    transcript=transcript,
                    quality_score=quality,
                )
            )

    with session_scope() as s:
        v = DatasetVersion(
            dataset_id=dataset_id,
            version="0.1.0",
            manifest_uri=f"file://{manifest_root}",
            num_samples=samples,
            total_audio_s=float(sum(durations) if durations else samples * 2.0),
        )
        s.add(v)
        s.flush()
        return dataset_id, v.id, manifest_root


def test_filter_by_duration() -> None:
    # 5 samples: durations = 1s, 2s, 3s, 4s, 5s
    dataset_id, version_id, _ = _seed_version(samples=5, durations=[1.0, 2.0, 3.0, 4.0, 5.0])
    with _client() as c:
        # Keep samples between 2.0s and 4.0s (should keep 2.0, 3.0, 4.0 -> 3 samples)
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/filter",
            json={
                "version": "0.2.0",
                "min_duration_s": 2.0,
                "max_duration_s": 4.0,
                "notes": "duration filtered",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["version"] == "0.2.0"
        assert body["num_samples"] == 3

        fork_id = body["id"]
        page = c.get(f"/datasets/versions/{fork_id}/samples?offset=0&limit=10").json()
        assert page["total"] == 3
        # Should contain samples with durations 2.0, 3.0, 4.0
        sample_durations = [item["audio"]["duration_s"] for item in page["items"]]
        assert sorted(sample_durations) == [2.0, 3.0, 4.0]


def test_filter_by_snr_and_quality() -> None:
    # snrs: 10, 20, None, 30, 40
    # qualities: 0.1, 0.5, 0.9, None, 0.8
    dataset_id, version_id, _ = _seed_version(
        samples=5,
        snrs=[10.0, 20.0, None, 30.0, 40.0],
        quality_scores=[0.1, 0.5, 0.9, None, 0.8]
    )
    with _client() as c:
        # Keep samples with SNR >= 20.0 and Quality Score >= 0.5
        # Index 1: SNR=20, Quality=0.5 -> Keep
        # Index 2: SNR=None, Quality=0.9 -> Discard
        # Index 3: SNR=30, Quality=None -> Discard
        # Index 4: SNR=40, Quality=0.8 -> Keep
        # Expected: 2 samples
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/filter",
            json={
                "version": "0.2.0",
                "min_snr_db": 20.0,
                "min_quality_score": 0.5,
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["num_samples"] == 2


def test_filter_by_text_length() -> None:
    # Transcripts with lengths: 3, 6, 9, 12, 15 characters
    transcripts = ["one", "two-22", "three-333", "four-444-444", "five-555-555-555"]
    dataset_id, version_id, _ = _seed_version(samples=5, transcripts=transcripts)
    with _client() as c:
        # Keep character lengths between 5 and 10 (should keep "two-22" (6), "three-333" (9) -> 2 samples)
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/filter",
            json={
                "version": "0.2.0",
                "min_text_len": 5,
                "max_text_len": 10,
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["num_samples"] == 2


def test_filter_validation_bounds() -> None:
    dataset_id, version_id, _ = _seed_version(samples=3)
    with _client() as c:
        # Reject min > max
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/filter",
            json={
                "version": "0.2.0",
                "min_duration_s": 5.0,
                "max_duration_s": 2.0,
            },
        )
        assert r.status_code == 400
        assert "min_duration_s must be <= max_duration_s" in r.text

        # Reject min text > max text
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/filter",
            json={
                "version": "0.2.0",
                "min_text_len": 10,
                "max_text_len": 5,
            },
        )
        assert r.status_code == 400
        assert "min_text_len must be <= max_text_len" in r.text
