"""Tests for dataset splits assignment.

Verifies random and speaker-disjoint splitting strategies via the API,
asserting that splits percentages and speaker boundaries are correctly respected.
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


def _seed_version(samples: int = 10, speaker_pattern: list[str] | None = None) -> tuple[str, str, Path]:
    """Create a Project, Dataset (ASR), and one DatasetVersion.

    If speaker_pattern is provided, it should be a list of speaker_ids of length `samples`.
    Returns ``(dataset_id, version_id, manifest_root)``.
    """
    from oas_core.db import Dataset, DatasetVersion, Project, init_db, session_scope
    from oas_core.settings import get_settings

    init_db()
    settings = get_settings()
    with session_scope() as s:
        p = Project(slug="splits-test", name="Splits Test")
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
        name="splits:0.1.0",
        modality=Modality.ASR,
        license_default=LicenseInfo(spdx="CC-BY-4.0"),
    )
    with ManifestWriter(manifest_root, header) as w:
        for i in range(samples):
            speaker_id = speaker_pattern[i] if speaker_pattern else None
            w.add(
                ASRSample(
                    license=LicenseInfo(spdx="CC-BY-4.0"),
                    audio=AudioRef(
                        uri=f"file:///fake/{i}.wav", sample_rate=16000, duration_s=1.0
                    ),
                    transcript=f"sample {i}",
                    speaker_id=speaker_id,
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


def test_splits_random_strategy() -> None:
    dataset_id, version_id, _ = _seed_version(samples=10)
    with _client() as c:
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/split",
            json={
                "version": "0.2.0",
                "train_pct": 0.6,
                "val_pct": 0.2,
                "test_pct": 0.2,
                "holdout_pct": 0.0,
                "strategy": "random",
                "notes": "split version",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["version"] == "0.2.0"
        assert body["parent_version_id"] == version_id
        assert body["num_samples"] == 10
        assert body["notes"] == "split version"

        # Check sample splits distribution
        fork_id = body["id"]
        page = c.get(f"/datasets/versions/{fork_id}/samples?offset=0&limit=10").json()
        assert page["total"] == 10
        
        splits = [item["split"] for item in page["items"]]
        train_count = splits.count("train")
        val_count = splits.count("val")
        test_count = splits.count("test")
        holdout_count = splits.count("holdout")

        # Expect exactly 60% (6), 20% (2), 20% (2)
        assert train_count == 6
        assert val_count == 2
        assert test_count == 2
        assert holdout_count == 0


def test_splits_speaker_disjoint_strategy() -> None:
    # 10 samples total: 4 for spk_a, 4 for spk_b, 2 for spk_c
    speakers = ["spk_a"] * 4 + ["spk_b"] * 4 + ["spk_c"] * 2
    dataset_id, version_id, _ = _seed_version(samples=10, speaker_pattern=speakers)
    with _client() as c:
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/split",
            json={
                "version": "0.3.0",
                "train_pct": 0.6,
                "val_pct": 0.2,
                "test_pct": 0.2,
                "holdout_pct": 0.0,
                "strategy": "speaker_disjoint",
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["num_samples"] == 10

        fork_id = body["id"]
        page = c.get(f"/datasets/versions/{fork_id}/samples?offset=0&limit=10").json()
        assert page["total"] == 10

        # Group actual split values by speaker
        spk_splits: dict[str, set[str]] = {}
        for item in page["items"]:
            spk = item["speaker_id"]
            split = item["split"]
            spk_splits.setdefault(spk, set()).add(split)

        # Assert speaker-disjoint constraint: each speaker must belong to exactly one split
        for spk, splits in spk_splits.items():
            assert len(splits) == 1, f"Speaker {spk} is split across multiple splits: {splits}"


def test_splits_validates_percentages_sum_to_one() -> None:
    dataset_id, version_id, _ = _seed_version(samples=5)
    with _client() as c:
        # Ratios do not sum to 1.0 (sums to 0.7)
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/split",
            json={
                "version": "0.2.0",
                "train_pct": 0.5,
                "val_pct": 0.1,
                "test_pct": 0.1,
            },
        )
        assert r.status_code == 400
        assert "must sum to 1.0" in r.text


def test_splits_handles_target_version_collision() -> None:
    dataset_id, version_id, parent_root = _seed_version(samples=5)
    
    # Pre-create 0.2.0 directory to force conflict
    collision = parent_root.parent / "0.2.0"
    collision.mkdir(parents=True, exist_ok=True)
    (collision / "manifest.json").write_text("{}")

    with _client() as c:
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/split",
            json={
                "version": "0.2.0",
                "train_pct": 0.8,
                "val_pct": 0.2,
                "test_pct": 0.0,
            },
        )
        assert r.status_code == 409
