"""Ray-based JobBackend.

Implements the same `JobBackend` interface as the in-process `WorkerPool` but
submits each job to a Ray cluster as a remote actor. The worker actor reads the
job row from the studio DB, runs the registered handler, and writes the Run
back — so server-side bookkeeping is identical regardless of backend.

This is opt-in: install `pip install 'oas-core[ray]'` and start the server
with `OAS_JOB_BACKEND=ray`.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

from oas_core.db import Job, JobStatus, Run, RunStatus, session_scope
from oas_core.queue.backend import JobBackend, JobContext, get_handler
from oas_core.settings import get_settings

log = logging.getLogger(__name__)


def _execute_job(job_id: str) -> dict[str, Any]:
    """Top-level executor — runs inside a Ray worker process."""
    settings = get_settings()
    with session_scope() as s:
        job = s.get(Job, job_id)
        if not job:
            raise RuntimeError(f"job {job_id!r} disappeared before execution")
        kind = job.kind
        config = dict(job.config)
        run = Run(job_id=job_id, attempt=1, status=RunStatus.RUNNING)
        run.started_at = datetime.now(UTC)
        s.add(run)
        s.flush()
        run_id = run.id

    artifacts_dir = settings.runs_dir / run_id / "artifacts"
    logs_dir = settings.runs_dir / run_id / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "stdout.log"

    def _sink(line: str) -> None:
        with log_file.open("a") as f:
            f.write(line + "\n")

    ctx = JobContext(
        job_id=job_id,
        run_id=run_id,
        kind=kind,
        config=config,
        artifacts_dir=str(artifacts_dir),
        logs_dir=str(logs_dir),
    )
    ctx.add_log_sink(_sink)

    metrics: dict[str, Any] = {}
    error: str | None = None
    try:
        handler = get_handler(kind)
        ctx.log(f"[ray] starting job kind={kind}")
        result = handler(ctx)
        if isinstance(result, dict):
            metrics = result
        status = RunStatus.SUCCEEDED
        job_status = JobStatus.SUCCEEDED
    except Exception as e:
        import traceback

        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        ctx.log(f"[ray] failed: {error}")
        status = RunStatus.FAILED
        job_status = JobStatus.FAILED

    finished = datetime.now(UTC)
    with session_scope() as s:
        db_run = s.get(Run, run_id)
        assert db_run is not None
        db_run.status = status
        db_run.metrics = metrics
        db_run.finished_at = finished
        db_run.logs_uri = f"file://{log_file}"
        db_run.artifacts_uri = f"file://{artifacts_dir}"
        db_run.error = error
        job = s.get(Job, job_id)
        assert job is not None
        job.status = job_status
        job.finished_at = finished
        job.error = error

    return {"status": status.value, "metrics": metrics}


class RayBackend(JobBackend):
    """Submits jobs to a Ray cluster."""

    def __init__(self, address: str | None = None, num_cpus: float | None = None) -> None:
        import ray  # type: ignore

        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=True)
        self.ray = ray
        self.num_cpus = num_cpus
        self._handles: dict[str, Any] = {}

    def start(self) -> None:
        # Ray actors are spun up per-job in submit(); nothing to do here.
        pass

    def stop(self, timeout: float = 5.0) -> None:
        # Leave the Ray cluster running — caller manages cluster lifecycle.
        pass

    def submit(self, job_id: str) -> None:
        with session_scope() as s:
            job = s.get(Job, job_id)
            if not job:
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)

        remote_kwargs: dict[str, Any] = {}
        if self.num_cpus is not None:
            remote_kwargs["num_cpus"] = self.num_cpus
        remote = self.ray.remote(**remote_kwargs)(_execute_job) if remote_kwargs else self.ray.remote(_execute_job)
        self._handles[job_id] = remote.remote(job_id)

    def cancel(self, job_id: str) -> None:
        handle = self._handles.pop(job_id, None)
        if handle is not None:
            with contextlib.suppress(Exception):
                self.ray.cancel(handle, force=True)
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                job.status = JobStatus.CANCELED
                job.finished_at = datetime.now(UTC)
