"""Slurm job backend.

Each studio job becomes an sbatch submission that runs a tiny Python entrypoint
in the cluster's environment. The entrypoint imports `oas_core` and calls
`_execute_job(job_id)`, which performs the same DB bookkeeping the in-process
backend uses.

Assumptions for v1:
- `sbatch` is on the studio host's PATH.
- The cluster mounts the studio's DB + storage (or both point at network
  services). Bare SQLite + local FS work only if the studio host *is* a Slurm
  login node with shared scratch.
- The cluster nodes have a Python environment with `oas-core` installed.
  Override via `SBATCH_PYTHON_ENV` / `OAS_SLURM_PARTITION` etc.

Environment overrides:
  OAS_SLURM_PARTITION     -> --partition
  OAS_SLURM_GPUS          -> --gres=gpu:<value>
  OAS_SLURM_TIME          -> --time
  OAS_SLURM_CPUS          -> --cpus-per-task
  OAS_SLURM_MEM           -> --mem
  OAS_SLURM_EXTRA         -> arbitrary extra sbatch flags
  OAS_SLURM_PYTHON        -> python interpreter on the node
  OAS_SLURM_SETUP         -> shell snippet to run before python (module load, source venv, etc.)
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from datetime import UTC, datetime
from textwrap import dedent

from oas_core.db import Job, JobStatus, session_scope
from oas_core.queue.backend import JobBackend
from oas_core.settings import get_settings

log = logging.getLogger(__name__)


ENTRY_TEMPLATE = dedent(
    """
    {setup}
    {python} -c '
    import sys
    from oas_core.queue.ray_backend import _execute_job
    res = _execute_job({job_id!r})
    print("[oas] done:", res)
    '
    """
).strip()


class SlurmBackend(JobBackend):
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.scripts_dir = settings.runs_dir / "_sbatch"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self._slurm_ids: dict[str, str] = {}

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

        script_path = self.scripts_dir / f"{job_id}.sh"
        log_path = self.scripts_dir / f"{job_id}.out"

        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=oas-{job_id}",
            f"#SBATCH --output={log_path}",
        ]
        for env_key, flag in (
            ("OAS_SLURM_PARTITION", "--partition"),
            ("OAS_SLURM_TIME", "--time"),
            ("OAS_SLURM_CPUS", "--cpus-per-task"),
            ("OAS_SLURM_MEM", "--mem"),
        ):
            val = os.environ.get(env_key)
            if val:
                sbatch_lines.append(f"#SBATCH {flag}={val}")
        gpus = os.environ.get("OAS_SLURM_GPUS")
        if gpus:
            sbatch_lines.append(f"#SBATCH --gres=gpu:{gpus}")
        extra = os.environ.get("OAS_SLURM_EXTRA")
        if extra:
            sbatch_lines.append(f"#SBATCH {extra}")

        python = os.environ.get("OAS_SLURM_PYTHON", "python3")
        setup = os.environ.get("OAS_SLURM_SETUP", "")
        body = ENTRY_TEMPLATE.format(setup=setup, python=python, job_id=job_id)

        script = "\n".join(sbatch_lines) + "\n\n" + body + "\n"
        script_path.write_text(script)
        script_path.chmod(0o755)

        try:
            out = subprocess.check_output(["sbatch", str(script_path)], text=True, timeout=30)
            # Output is "Submitted batch job 12345"
            slurm_id = out.strip().split()[-1]
            self._slurm_ids[job_id] = slurm_id
            log.info("sbatch submitted job=%s slurm_id=%s", job_id, slurm_id)
        except Exception as e:
            with session_scope() as s:
                job = s.get(Job, job_id)
                if job:
                    job.status = JobStatus.FAILED
                    job.finished_at = datetime.now(UTC)
                    job.error = f"sbatch failed: {e}"
            raise

    def cancel(self, job_id: str) -> None:
        slurm_id = self._slurm_ids.pop(job_id, None)
        if slurm_id:
            with contextlib.suppress(Exception):
                subprocess.run(["scancel", slurm_id], check=False, timeout=10)
        with session_scope() as s:
            job = s.get(Job, job_id)
            if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                job.status = JobStatus.CANCELED
                job.finished_at = datetime.now(UTC)
