import contextlib
import time

from oas_core.db import Job, JobStatus, Run, RunStatus, init_db, session_scope
from oas_core.queue import WorkerPool, register_handler, submit_job
from oas_core.queue.backend import JobContext


def _echo_handler(ctx: JobContext) -> dict:
    ctx.log("hello from job")
    return {"echoed": ctx.config.get("msg", "")}


def _failing_handler(ctx: JobContext) -> dict:
    raise RuntimeError("boom")


def test_job_runs_and_succeeds(project_id) -> None:
    with contextlib.suppress(ValueError):
        register_handler("echo", _echo_handler)
    init_db()
    pool = WorkerPool(concurrency=1, poll_interval_s=0.05)
    pool.start()
    try:
        job_id = submit_job(project_id, "echo", "test-echo", {"msg": "hi"})
        deadline = time.time() + 5
        status = None
        while time.time() < deadline:
            with session_scope() as s:
                job = s.get(Job, job_id)
                status = job.status if job else None
                if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                    break
            time.sleep(0.05)
        assert status == JobStatus.SUCCEEDED
        with session_scope() as s:
            runs = s.query(Run).filter(Run.job_id == job_id).all()
            assert len(runs) == 1
            assert runs[0].status == RunStatus.SUCCEEDED
            assert runs[0].metrics == {"echoed": "hi"}
    finally:
        pool.stop()


def test_job_failure_recorded(project_id) -> None:
    with contextlib.suppress(ValueError):
        register_handler("fail", _failing_handler)
    init_db()
    pool = WorkerPool(concurrency=1, poll_interval_s=0.05)
    pool.start()
    try:
        job_id = submit_job(project_id, "fail", "test-fail", {})
        deadline = time.time() + 5
        status = None
        while time.time() < deadline:
            with session_scope() as s:
                job = s.get(Job, job_id)
                status = job.status if job else None
                if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                    break
            time.sleep(0.05)
        assert status == JobStatus.FAILED
        with session_scope() as s:
            job = s.get(Job, job_id)
            assert job and job.error and "boom" in job.error
    finally:
        pool.stop()
