"""Tests for dataset deduplication.

Verifies deduplication by exact text matching, audio hash matching,
and token-based similar text matching via the API.
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


def _seed_version(samples: int = 5, transcripts: list[str] | None = None, hashes: list[str | None] | None = None) -> tuple[str, str, Path]:
    """Create a Project, Dataset (ASR), and one DatasetVersion with customizable features."""
    from oas_core.db import Dataset, DatasetVersion, Project, init_db, session_scope
    from oas_core.settings import get_settings

    init_db()
    settings = get_settings()
    with session_scope() as s:
        p = Project(slug="dedup-test", name="Dedup Test")
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
        name="dedup:0.1.0",
        modality=Modality.ASR,
        license_default=LicenseInfo(spdx="CC-BY-4.0"),
    )
    with ManifestWriter(manifest_root, header) as w:
        for i in range(samples):
            transcript = transcripts[i] if transcripts else f"sample {i}"
            sha256 = hashes[i] if hashes else None
            w.add(
                ASRSample(
                    license=LicenseInfo(spdx="CC-BY-4.0"),
                    audio=AudioRef(
                        uri=f"file:///fake/{i}.wav", sample_rate=16000, duration_s=1.0, sha256=sha256
                    ),
                    transcript=transcript,
                )
            )

    with session_scope() as s:
        v = DatasetVersion(
            dataset_id=dataset_id,
            version="0.1.0",
            manifest_uri=f"file://{manifest_root}",
            num_samples=samples,
            total_audio_s=float(samples),
        )
        s.add(v)
        s.flush()
        return dataset_id, v.id, manifest_root


def test_dedup_exact_text() -> None:
    # 5 samples, 2 pairs of duplicate transcripts (case-insensitive, whitespace padded)
    transcripts = [
        "Hello world",
        "  hello WORLD ",  # Duplicate of 0
        "Distinct text",
        "Another distinct",
        "hello World"       # Duplicate of 0
    ]
    dataset_id, version_id, _ = _seed_version(samples=5, transcripts=transcripts)
    with _client() as c:
        # Exact text strategy should leave exactly 3 unique samples
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/dedup",
            json={
                "version": "0.2.0",
                "strategy": "exact_text",
                "notes": "exact text deduped",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["version"] == "0.2.0"
        assert body["num_samples"] == 3

        fork_id = body["id"]
        page = c.get(f"/datasets/versions/{fork_id}/samples?offset=0&limit=10").json()
        assert page["total"] == 3
        # Should keep "Hello world", "Distinct text", "Another distinct"
        texts = [item["transcript"].strip() for item in page["items"]]
        assert "Hello world" in texts
        assert "Distinct text" in texts
        assert "Another distinct" in texts


def test_dedup_audio_hash() -> None:
    # 5 samples: identical audio hashes for index 0, 1, 3
    hashes = ["hash_a", "hash_a", "hash_b", "hash_a", None]
    dataset_id, version_id, _ = _seed_version(samples=5, hashes=hashes)
    with _client() as c:
        # Should retain:
        # Index 0 ("hash_a") -> Keep
        # Index 1 ("hash_a") -> Discard
        # Index 2 ("hash_b") -> Keep
        # Index 3 ("hash_a") -> Discard
        # Index 4 (None) -> Keep (missing hash, not duplicate of anything)
        # Expected: 3 samples
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/dedup",
            json={
                "version": "0.2.0",
                "strategy": "audio_hash",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["num_samples"] == 3


def test_dedup_similar_text() -> None:
    transcripts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumped over the lazy dog",  # Jaccard = 8/10 = 0.8 (extremely similar)
        "completely different unrelated sentences and words", # Keep
        "lazy dog jumps over the quick brown fox", # Jaccard = 100% (same words shuffled)
    ]
    dataset_id, version_id, _ = _seed_version(samples=4, transcripts=transcripts)
    with _client() as c:
        # Similar text strategy with threshold 0.75:
        # Index 0 -> Keep
        # Index 1 (Jaccard w/ 0 = 8/10 = 0.8 >= 0.75) -> Discard
        # Index 2 (completely different) -> Keep
        # Index 3 (Jaccard w/ 0 = 1.0 >= 0.75) -> Discard
        # Expected: 2 samples
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/dedup",
            json={
                "version": "0.2.0",
                "strategy": "similar_text",
                "threshold": 0.75,
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["num_samples"] == 2
