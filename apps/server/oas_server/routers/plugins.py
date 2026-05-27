"""Plugins SDK: dynamic registry, live PIP installer, and custom package scaffolding."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
from importlib.metadata import entry_points, metadata
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from oas_core.db import Role
from oas_core.queue.backend import list_handlers, reset_plugin_discovery
from oas_core.settings import get_settings
from pydantic import BaseModel, Field

from oas_server.auth import CurrentUser, assert_role, require_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/plugins", tags=["plugins"])


class PluginOut(BaseModel):
    name: str
    version: str
    description: str | None
    handlers: list[str]
    code_path: str | None


class InstallIn(BaseModel):
    source: str = Field(description="Local folder path, PyPI name, or Git URL to install")


class InstallOut(BaseModel):
    success: bool
    returncode: int
    logs: str


class ScaffoldIn(BaseModel):
    name: str = Field(min_length=1, max_length=100, description="Plugin name, e.g. 'mimi-encoder-v2'")
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$", description="Unique job kind identifier")
    description: str | None = None


class ScaffoldOut(BaseModel):
    success: bool
    slug: str
    destination: str
    message: str


@router.get("", response_model=list[PluginOut])
def list_plugins(user: CurrentUser = Depends(require_user)) -> list[PluginOut]:
    # Superusers or project members can view plugins
    # Discovery reset first so newly hot-installed packages load immediately
    reset_plugin_discovery()
    list_handlers()

    try:
        eps = entry_points(group="oas.handlers")
    except Exception:
        return []

    plugins_dict: dict[str, dict[str, Any]] = {}
    for ep in eps:
        dist_name = ep.dist.name if ep.dist else "built-in"
        if dist_name not in plugins_dict:
            version = "0.1.0"
            desc = None
            if ep.dist:
                with contextlib.suppress(Exception):
                    meta = metadata(dist_name)
                    version = meta.get("Version", "0.1.0")
                    desc = meta.get("Summary")
            plugins_dict[dist_name] = {
                "name": dist_name,
                "version": version,
                "description": desc,
                "handlers": [],
                "code_path": ep.value if hasattr(ep, "value") else str(ep),
            }
        plugins_dict[dist_name]["handlers"].append(ep.name)

    return [PluginOut(**p) for p in plugins_dict.values()]


@router.post("/install", response_model=InstallOut)
def install_plugin(
    body: InstallIn,
    user: CurrentUser = Depends(require_user),
) -> InstallOut:
    # Only superusers can install python packages into the server environment for security!
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Plugin installations require server superuser privilege.",
        )

    # Use sys.executable to guarantee installing in the exact same virtual environment
    # -m pip install is the most robust installation method.
    cmd = [sys.executable, "-m", "pip", "install"]

    # Support local dev editable installations if source starts with / or is a directory
    source_path = Path(body.source)
    if source_path.is_absolute() and source_path.exists():
        cmd.extend(["-e", str(source_path)])
    else:
        cmd.append(body.source)

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        logs = f"Stdout:\n{res.stdout}\n\nStderr:\n{res.stderr}"
        success = res.returncode == 0

        if success:
            # Clear discovery cache and re-run entrypoints discovery to hot-reload handlers
            reset_plugin_discovery()
            list_handlers()
            log.info("Successfully installed plugin and hot-loaded handlers from %s", body.source)

        return InstallOut(success=success, returncode=res.returncode, logs=logs)
    except Exception as e:
        return InstallOut(
            success=False,
            returncode=-1,
            logs=f"Failed to execute pip install: {e}",
        )


@router.post("/scaffold", response_model=ScaffoldOut)
def scaffold_plugin(
    body: ScaffoldIn,
    user: CurrentUser = Depends(require_user),
) -> ScaffoldOut:
    # Requires at least EDITOR role on a project to scaffold a plugin (we check for a default admin setting or superuser)
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Scaffolding requires superuser privilege.",
        )

    settings = get_settings()
    slug = re.sub(r"[^a-z0-9_-]", "", body.name.lower().replace(" ", "-"))
    if not slug:
        raise HTTPException(400, "Invalid plugin name")

    pkg_dir = f"oas_{slug.replace('-', '_')}"
    dest_dir = settings.data_dir.parent / "plugins" / slug
    dest_dir.mkdir(parents=True, exist_ok=True)

    pyproject_content = f"""[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "oas-{slug}"
version = "0.1.0"
description = "{body.description or 'Custom job handler plugin for Open Audio Studio'}"
requires-python = ">=3.11"
dependencies = ["oas-core"]

[project.entry-points."oas.handlers"]
{body.kind} = "{pkg_dir}.handlers:{body.kind}_handler"

[tool.setuptools.packages.find]
include = ["{pkg_dir}*"]
"""

    # Create package directories
    pkg_path = dest_dir / pkg_dir
    pkg_path.mkdir(exist_ok=True)

    init_content = '"""Open Audio Studio custom plugin package."""\n'
    handlers_content = f"""\"\"\"Custom job handler implementation.

This callable represents the entry point for the '{body.kind}' job type.
\"\"\"

import logging
from datetime import UTC, datetime
from oas_core.db import Job, JobStatus, session_scope

log = logging.getLogger(__name__)


def {body.kind}_handler(job_id: str) -> dict:
    \"\"\"Boilerplate runner representing execution of a '{body.kind}' task.

    This function is executed by the WorkerPool in a background thread or process.
    All database updates and bookkeeping for Job runs are handled automatically.
    \"\"\"
    log.info("Initializing custom '{body.kind}' job: %s", job_id)

    with session_scope() as s:
        job = s.get(Job, job_id)
        if not job:
            raise KeyError(f"Job %s not found in registry" % job_id)
        
        # Access configurations passed by the user from settings
        config = job.config or {{}}
        log.info("Loaded custom configuration: %s", config)

    # Simulated training iteration loop
    log.info("Starting training optimization...")
    for step in range(5):
        import time
        time.sleep(0.5)
        log.info("Step %d/5: training reconstructions loss = %f", step + 1, 0.45 - (step * 0.08))

    log.info("Custom job execution completed successfully.")
    return {{
        "status": "success",
        "finished_at": datetime.now(UTC).isoformat(),
        "final_loss": 0.13,
        "exported_checkpoint": "oas-{slug}-v1.0"
    }}
"""

    (dest_dir / "pyproject.toml").write_text(pyproject_content)
    (pkg_path / "__init__.py").write_text(init_content)
    (pkg_path / "handlers.py").write_text(handlers_content)

    return ScaffoldOut(
        success=True,
        slug=slug,
        destination=str(dest_dir),
        message=f"Plugin scaffolded successfully in '{dest_dir}'. To install, run: pip install -e {dest_dir}",
    )
