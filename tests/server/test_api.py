from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _client() -> TestClient:
    # Import inside the function so test fixtures can set env first.
    from oas_server.main import create_app

    return TestClient(create_app())


def test_health_and_system() -> None:
    with _client() as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        info = c.get("/system/info").json()
        assert "version" in info
        assert "noop" in info["handlers"]


def test_project_and_dataset_crud() -> None:
    with _client() as c:
        r = c.post("/projects", json={"slug": "demo", "name": "Demo"})
        assert r.status_code == 201, r.text
        pid = r.json()["id"]

        r = c.post(
            "/datasets",
            json={
                "project_id": pid,
                "slug": "asr1",
                "name": "ASR 1",
                "modality": "asr",
            },
        )
        assert r.status_code == 201, r.text
        did = r.json()["id"]

        r = c.post(
            f"/datasets/{did}/versions",
            json={
                "version": "0.1.0",
                "manifest_uri": "file:///tmp/m",
                "num_samples": 5,
                "total_audio_s": 12.5,
            },
        )
        assert r.status_code == 201, r.text

        versions = c.get(f"/datasets/{did}/versions").json()
        assert len(versions) == 1
        assert versions[0]["num_samples"] == 5


def test_job_submission_runs_to_completion() -> None:
    with _client() as c:
        pid = c.post("/projects", json={"slug": "jobs", "name": "Jobs"}).json()["id"]
        r = c.post(
            "/jobs",
            json={"project_id": pid, "kind": "noop", "name": "first-job", "config": {}},
        )
        assert r.status_code == 201, r.text
        job_id = r.json()["id"]

        deadline = time.time() + 5
        final = None
        while time.time() < deadline:
            j = c.get(f"/jobs/{job_id}").json()
            if j["status"] in ("succeeded", "failed"):
                final = j
                break
            time.sleep(0.1)
        assert final and final["status"] == "succeeded", final
        assert final["runs"] and final["runs"][0]["status"] == "succeeded"


def test_cancel_job_returns_404_when_row_disappears_mid_call() -> None:
    # The cancel route reloads the Job row after calling pool.cancel() so
    # the response reflects the post-cancel state. If the row is deleted
    # in that window — by a concurrent operator action or a cascading cleanup —
    # the reload must surface as a clean 404, not a 500 from None-deref or a
    # stale successful response.
    from oas_core.db import Job, session_scope
    from oas_server.main import create_app

    app = create_app()
    with TestClient(app) as c:
        pid = c.post("/projects", json={"slug": "race", "name": "Race"}).json()["id"]
        job_id = c.post(
            "/jobs",
            json={"project_id": pid, "kind": "noop", "name": "racy", "config": {}},
        ).json()["id"]

        # Simulate the race: pool.cancel deletes the job row, modelling a
        # concurrent delete that lands between the route's first lookup and
        # its post-cancel reload.
        pool = app.state.worker_pool

        def _delete_during_cancel(jid: str) -> None:
            with session_scope() as s:
                j = s.get(Job, jid)
                if j is not None:
                    s.delete(j)

        pool.cancel = _delete_during_cancel  # type: ignore[method-assign]

        r = c.post(f"/jobs/{job_id}/cancel")
        assert r.status_code == 404, r.text


def test_model_registry_via_api() -> None:
    with _client() as c:
        pid = c.post("/projects", json={"slug": "reg", "name": "Reg"}).json()["id"]
        m = c.post(
            "/models",
            json={
                "project_id": pid,
                "slug": "whisper-en",
                "name": "Whisper EN",
                "modality": "asr",
            },
        ).json()
        v = c.post(
            f"/models/{m['id']}/versions",
            json={
                "version": "0.1.0",
                "artifact_uri": "file:///tmp/x",
                "metrics": {"wer": 0.1},
            },
        ).json()
        assert v["stage"] == "dev"
        v2 = c.post(f"/models/versions/{v['id']}/stage", params={"stage": "staging"}).json()
        assert v2["stage"] == "staging"
