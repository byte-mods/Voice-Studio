"""End-to-end smoke test of the studio's primary flow.

Exercises: signup → project (creator becomes admin) → dataset → manifest write
on disk → DatasetVersion registered → model registered → registry version
published with metrics → eval-results endpoint reachable → audit log captured.

Does not run real ML; uses noop / pre-built manifests so the test stays fast
and GPU-free.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


def test_full_studio_flow() -> None:
    with _client() as c:
        # 1) Signup (becomes superuser since first user)
        r = c.post(
            "/auth/signup",
            json={"email": "e2e@x.com", "password": "password123", "name": "E2E"},
        )
        assert r.status_code == 201, r.text
        token = r.json()["access_token"]
        h = {"authorization": f"Bearer {token}"}

        # 2) Create a project
        proj = c.post("/projects", json={"slug": "e2e", "name": "E2E"}, headers=h).json()

        # 3) Create dataset + version
        ds = c.post(
            "/datasets",
            json={
                "project_id": proj["id"],
                "slug": "asr-e2e",
                "name": "ASR E2E",
                "modality": "asr",
            },
            headers=h,
        ).json()

        from oas_core.manifest import (
            ASRSample,
            AudioRef,
            LicenseInfo,
            ManifestHeader,
            ManifestWriter,
            Modality,
        )
        from oas_core.settings import get_settings

        s = get_settings()
        manifest_root = s.datasets_dir / ds["id"] / "0.1.0"
        manifest_root.mkdir(parents=True, exist_ok=True)
        header = ManifestHeader(dataset_id=ds["id"], name="e2e", modality=Modality.ASR)
        with ManifestWriter(manifest_root, header) as w:
            w.add(
                ASRSample(
                    license=LicenseInfo(spdx="CC0-1.0"),
                    audio=AudioRef(uri="file:///fake.wav", sample_rate=16000, duration_s=1.0),
                    transcript="hello",
                )
            )

        v = c.post(
            f"/datasets/{ds['id']}/versions",
            json={
                "version": "0.1.0",
                "manifest_uri": f"file://{manifest_root}",
                "num_samples": 1,
                "total_audio_s": 1.0,
            },
            headers=h,
        ).json()
        assert v["num_samples"] == 1

        # 4) Submit a noop job and wait
        job = c.post(
            "/jobs",
            json={
                "project_id": proj["id"],
                "kind": "noop",
                "name": "e2e-noop",
                "config": {},
            },
            headers=h,
        ).json()
        deadline = time.time() + 5
        while time.time() < deadline:
            jr = c.get(f"/jobs/{job['id']}", headers=h).json()
            if jr["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert jr["status"] == "succeeded"

        # 5) Register model + publish a version
        m = c.post(
            "/models",
            json={
                "project_id": proj["id"],
                "slug": "whisper-e2e",
                "name": "Whisper E2E",
                "modality": "asr",
            },
            headers=h,
        ).json()
        mv = c.post(
            f"/models/{m['id']}/versions",
            json={
                "version": "0.1.0",
                "artifact_uri": "file:///fake/ckpt",
                "metrics": {"wer": 0.12},
            },
            headers=h,
        ).json()
        assert mv["stage"] == "dev"

        # 6) Eval-results endpoint is reachable (empty list expected)
        evals = c.get(f"/models/versions/{mv['id']}/evals", headers=h).json()
        assert isinstance(evals, list)

        # 7) Audit log captured at least one of our mutations
        log = c.get("/audit", headers=h).json()
        assert any(entry["path"] == "/projects" and entry["method"] == "POST" for entry in log)
        assert any(entry["actor_email"] == "e2e@x.com" for entry in log)
