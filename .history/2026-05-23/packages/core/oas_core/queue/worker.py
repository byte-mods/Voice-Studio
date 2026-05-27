"""In-process worker pool.

The pool polls the DB for QUEUED jobs, claims them with a row-level update,
runs them on a thread, and records the resulting Run. It is intentionally
simple — production deployments will swap this for a Ray/Slurm/k8s backend
implementing the same `JobBackend` interface.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update

from oas_core.db import Job, JobStatus, Run, RunStatus, session_scope
from oas_core.queue.backend import JobBackend, JobContext, get_handler
from oas_core.settings import get_settings

log = logging.getLogger(__name__)


def submit_job(
    project_id: str,
    kind: str,
    name: str,
    config: Mapping[str, Any],
    *,
    priority: int = 0,
) -> str:
    """Persist a job in QUEUED status and return its id."""
    with session_scope() as s:
        job = Job(
            project_id=project_id,
            kind=kind,
            name=name,
            config=dict(config),
            priority=priority,
        )
        s.add(job)
        s.flush()
        return job.id


class WorkerPool(JobBackend):
    """A daemon thread pool that drains the job queue."""

    def __init__(self, concurrency: int | None = None, poll_interval_s: float = 1.0) -> None:
        settings = get_settings()
        self.concurrency = concurrency or settings.worker_concurrency
        self.poll_interval_s = poll_interval_s
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._cancelled_jobs: set[str] = set()
        self._lock = threading.Lock()

    # ---- JobBackend interface ----

    def submit(self, job_id: str) -> None:
        # In-process backend reads from the DB on its own; no-op.
        pass

    def cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancelled_jobs.add(job_id)
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job and job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELED
                job.finished_at = datetime.now(UTC)

    # ---- Lifecycle ----

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()
        for i in range(self.concurrency):
            t = threading.Thread(target=self._loop, name=f"oas-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("WorkerPool started with %d workers", self.concurrency)

    def stop(self, timeout: float = 30.0) -> None:
        """Graceful drain: stop accepting new claims, wait up to `timeout`
        seconds for in-flight jobs to finish, then join workers."""
        self._stop.set()
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._inflight_lock:
                if not self._inflight:
                    break
            time.sleep(0.1)
        with self._inflight_lock:
            remaining = list(self._inflight)
        if remaining:
            log.warning(
                "WorkerPool draining timed out with %d job(s) still running: %s",
                len(remaining),
                remaining,
            )
        for t in self._threads:
            t.join(timeout=max(0.5, deadline - time.time()))
        self._threads.clear()

    # ---- Internals ----

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._claim_one()
                if job_id is None:
                    self._stop.wait(self.poll_interval_s)
                    continue
                self._run_job(job_id)
            except Exception:
                log.exception("Worker loop error")
                self._stop.wait(self.poll_interval_s)

    def _claim_one(self) -> str | None:
        """Atomically transition one QUEUED job to RUNNING and return its id."""
        with session_scope() as s:
            stmt = (
                select(Job.id)
                .where(Job.status == JobStatus.QUEUED)
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            row = s.execute(stmt).first()
            if row is None:
                return None
            job_id: str = row[0]
            now = datetime.now(UTC)
            s.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == JobStatus.QUEUED)
                .values(status=JobStatus.RUNNING, started_at=now)
            )
            return job_id

    def _run_job(self, job_id: str) -> None:
        settings = get_settings()
        with self._inflight_lock:
            self._inflight.add(job_id)
        with session_scope() as s:
            job = s.get(Job, job_id)
            assert job is not None
            kind = job.kind
            config = dict(job.config)

            run = Run(job_id=job.id, attempt=1, status=RunStatus.RUNNING)
            run.started_at = datetime.now(UTC)
            s.add(run)
            s.flush()
            run_id = run.id

        artifacts_dir = settings.runs_dir / run_id / "artifacts"
        logs_dir = settings.runs_dir / run_id / "logs"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "stdout.log"

        def _file_sink(line: str) -> None:
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
        ctx.add_log_sink(_file_sink)

        metrics: dict[str, Any] = {}
        error: str | None = None
        try:
            handler = get_handler(kind)
            ctx.log(f"Starting job kind={kind}")
            result = handler(ctx)
            if isinstance(result, dict):
                metrics = result
            ctx.log("Job finished")
            status = RunStatus.SUCCEEDED
            job_status = JobStatus.SUCCEEDED
        except Exception as e:
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            ctx.log(f"Job failed: {error}")
            status = RunStatus.FAILED
            job_status = JobStatus.FAILED
        finally:
            with self._inflight_lock:
                self._inflight.discard(job_id)

        finished = datetime.now(UTC)
        with session_scope() as s:
            # Re-fetch under a fresh session; row is guaranteed to exist (we just wrote it).
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


def _wait_until(predicate, timeout: float, interval: float = 0.05) -> bool:
    """Test helper: poll `predicate()` until True or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
