"""Lineage: forking a ``DatasetVersion`` copies its samples and records provenance.

The fork is a metadata + manifest copy. Audio refs are preserved by URI — no
file-level duplication. ``parent_version_id`` makes the provenance link
inspectable and queryable.
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


def _seed_version(samples: int = 3) -> tuple[str, str, Path]:
    """Create a Project, Dataset (ASR), and one DatasetVersion with ``samples`` samples.

    Returns ``(dataset_id, version_id, manifest_root)``.
    """
    from oas_core.db import Dataset, DatasetVersion, Project, init_db, session_scope
    from oas_core.settings import get_settings

    init_db()
    settings = get_settings()
    with session_scope() as s:
        p = Project(slug="lineage", name="Lineage")
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
        name="lineage:0.1.0",
        modality=Modality.ASR,
        license_default=LicenseInfo(spdx="CC-BY-4.0"),
    )
    with ManifestWriter(manifest_root, header) as w:
        for i in range(samples):
            w.add(
                ASRSample(
                    license=LicenseInfo(spdx="CC-BY-4.0"),
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
            num_samples=samples,
            total_audio_s=float(samples),
        )
        s.add(v)
        s.flush()
        return dataset_id, v.id, manifest_root


def test_fork_version_copies_samples_and_records_parent() -> None:
    dataset_id, version_id, parent_root = _seed_version(samples=3)
    with _client() as c:
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/fork",
            json={"version": "0.2.0", "notes": "forked for cleanup"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["version"] == "0.2.0"
        assert body["parent_version_id"] == version_id
        assert body["num_samples"] == 3
        assert body["notes"] == "forked for cleanup"
        # Manifest URI points at a fresh directory, not the parent's.
        assert body["manifest_uri"] != f"file://{parent_root}"

        # Samples are actually queryable through the standard preview endpoint.
        fork_id = body["id"]
        page = c.get(f"/datasets/versions/{fork_id}/samples?offset=0&limit=10").json()
        assert page["total"] == 3
        assert {item["transcript"] for item in page["items"]} == {
            "sample 0",
            "sample 1",
            "sample 2",
        }


def test_fork_returns_404_when_parent_missing() -> None:
    dataset_id, _, _ = _seed_version(samples=1)
    with _client() as c:
        r = c.post(
            f"/datasets/{dataset_id}/versions/does-not-exist/fork",
            json={"version": "0.2.0"},
        )
        assert r.status_code == 404


def test_fork_returns_400_when_parent_belongs_to_a_different_dataset() -> None:
    from oas_core.db import Dataset, Project, session_scope

    _, version_id_a, _ = _seed_version(samples=1)
    with session_scope() as s:
        # Same project, different dataset — fork must not cross dataset boundaries.
        p = s.execute(__import__("sqlalchemy").select(Project)).scalars().first()
        assert p is not None
        other = Dataset(project_id=p.id, slug="asr2", name="ASR2", modality=Modality.ASR)
        s.add(other)
        s.flush()
        dataset_id_b = other.id

    with _client() as c:
        r = c.post(
            f"/datasets/{dataset_id_b}/versions/{version_id_a}/fork",
            json={"version": "0.2.0"},
        )
        assert r.status_code == 400
        assert "different dataset" in r.text


def test_fork_returns_409_when_version_already_exists_on_disk() -> None:
    dataset_id, version_id, parent_root = _seed_version(samples=1)
    # Pre-create the target directory with a sentinel file to simulate collision.
    collision = parent_root.parent / "0.2.0"
    collision.mkdir(parents=True, exist_ok=True)
    (collision / "marker").write_text("squatter")

    with _client() as c:
        r = c.post(
            f"/datasets/{dataset_id}/versions/{version_id}/fork",
            json={"version": "0.2.0"},
        )
        assert r.status_code == 409
