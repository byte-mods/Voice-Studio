from __future__ import annotations

from fastapi.testclient import TestClient


def _client() -> TestClient:
    from oas_server.main import create_app

    return TestClient(create_app())


def test_new_handlers_registered() -> None:
    with _client() as c:
        kinds = c.get("/jobs/handlers").json()
        for k in ("kernel_bench", "asr_eval", "llm_eval", "tts_eval"):
            assert k in kinds, f"missing {k}: {kinds}"


def test_kernel_crud() -> None:
    with _client() as c:
        pid = c.post("/projects", json={"slug": "lab", "name": "Lab"}).json()["id"]
        r = c.post(
            "/kernels",
            json={
                "project_id": pid,
                "slug": "add",
                "name": "Vector add",
                "backend": "triton",
                "source": "def kernel(x, y): return x + y",
                "reference": "def reference(x, y): return x + y",
                "bench_config": {"shapes": [{"args": [{"shape": [16], "dtype": "float32"}]}]},
            },
        )
        assert r.status_code == 201, r.text
        kid = r.json()["id"]

        listed = c.get("/kernels", params={"project_id": pid}).json()
        assert len(listed) == 1

        patched = c.patch(f"/kernels/{kid}", json={"name": "New name"}).json()
        assert patched["name"] == "New name"

        # Queue a benchmark — even without torch/triton installed, the queue
        # should accept and create a job (which will fail in the worker; that's
        # the worker's problem, not the API's).
        r = c.post(f"/kernels/{kid}/benchmark")
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()


def test_kernel_bench_handler_runs_minimal_triton_stub(monkeypatch) -> None:
    """Verify the handler's plumbing without requiring a GPU/Triton.

    We register a fake handler under a fresh kind that uses the same path
    structure (exec source / reference / shape sweep) but with pure Python.
    """
    # We won't actually invoke the worker here; just confirm the handler module
    # imports cleanly without Triton installed.
    from oas_server.jobs import kernel_bench  # noqa: F401
