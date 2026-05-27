"""Modal job backend.

Submits each job to a Modal app as a remote function call. The remote function
re-enters the studio's `_execute_job` so all bookkeeping (`Run` row, log file,
artifacts dir, metrics) is identical to local execution. The only thing that
differs is *where* the handler runs.

Requires:
- `pip install 'oas-core[modal]'`
- A Modal token configured (`modal token new`)
- The studio DB and storage reachable from the Modal container. For a local
  studio that means using a shared Postgres / S3 — Modal cannot see your
  laptop's SQLite. The backend raises a clear error if the configured DB URL
  is sqlite.

This adapter intentionally constructs the Modal `App` lazily so importing the
module without `modal` installed is harmless.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

from oas_core.db import Job, JobStatus, session_scope
from oas_core.queue.backend import JobBackend
from oas_core.queue.ray_backend import _execute_job  # shared executor
from oas_core.settings import Settings, get_settings

log = logging.getLogger(__name__)


def _check_shared_state(settings: Settings) -> None:
    if settings.db_url.startswith("sqlite"):
        raise RuntimeError(
            "Modal backend requires a network-reachable DB (Postgres / MySQL). "
            "SQLite cannot be shared with Modal containers."
        )


class ModalBackend(JobBackend):
    def __init__(
        self,
        app_name: str = "open-audio-studio",
        image: str | None = None,
        timeout_s: int = 60 * 60,
        gpu: str | None = None,
    ) -> None:
        import modal

        settings = get_settings()
        _check_shared_state(settings)

        self.modal = modal
        self.app = modal.App(app_name)
        self.timeout_s = timeout_s
        self.gpu = gpu

        # Build a Modal image that includes oas-core. In production this should
        # be a pre-built image with the user's plugins; here we install from a
        # mounted local directory if available, otherwise from PyPI.
        self.image = (
            image
            if image is not None
            else modal.Image.debian_slim(python_version="3.11")
            .pip_install("oas-core")
        )

        @self.app.function(image=self.image, timeout=timeout_s, gpu=gpu)  # type: ignore[untyped-decorator]
        def _runner(job_id: str) -> dict[str, Any]:
            return _execute_job(job_id)

        self._runner = _runner
        self._handles: dict[str, Any] = {}

    def start(self) -> None:
        pass

    def stop(self, timeout: float = 5.0) -> None:
        pass

    def submit(self, job_id: str) -> None:
        with session_scope() as s:
            job = s.get(Job, job_id)
            if not job:
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)

        with self.app.run():
            self._handles[job_id] = self._runner.spawn(job_id)

    def cancel(self, job_id: str) -> None:
        handle = self._handles.pop(job_id, None)
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.cancel()
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                job.status = JobStatus.CANCELED
                job.finished_at = datetime.now(UTC)
