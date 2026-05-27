"""In-process job queue with a worker pool.

Design:
- Jobs are persisted in the `jobs` table; the queue is just a coordinator.
- A `JobHandler` registry maps `job.kind` -> callable that runs the job.
- The worker pool polls the DB for queued jobs, claims them atomically, creates
  a `Run`, executes the handler, and records metrics + status.
- This is a foundation. Phase 5+ adds Ray and Slurm adapters that implement
  the same `JobBackend` interface.
"""

from oas_core.queue.backend import JobBackend, JobContext, JobHandler, register_handler
from oas_core.queue.worker import WorkerPool, submit_job

__all__ = [
    "JobBackend",
    "JobContext",
    "JobHandler",
    "WorkerPool",
    "register_handler",
    "submit_job",
]
