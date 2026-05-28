"""Job handler: download a Hugging Face model repository and register it in OAS.

Config schema (`Job.config`):

    {
      "project_id": "<oas Project.id>",                   # required
      "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",              # required
      "modality": "llm",                                  # required
      "version": "1.0.0",                                 # optional, default '1.0.0'
      "family": "qwen"                                    # optional
    }
"""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import Any

from oas_core.db import Model, session_scope
from oas_core.db.models import Modality, ModelStage
from oas_core.queue.backend import JobContext, register_handler
from oas_core.settings import get_settings

log = logging.getLogger(__name__)


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"hf_model_download config missing required key: {key!r}")
    return d[key]


def hf_model_download_handler(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    project_id = _require(cfg, "project_id")
    hf_id = _require(cfg, "hf_id")
    modality_str = _require(cfg, "modality")
    version = cfg.get("version", "1.0.0")
    family = cfg.get("family")

    modality = Modality(modality_str)
    settings = get_settings()

    # Create a stable directory name for local cache
    safe_name = hf_id.replace("/", "_")
    dest_dir = settings.models_dir / safe_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    ctx.log(f"Starting Hugging Face model download for {hf_id!r}...")
    ctx.log(f"Destination directory: {dest_dir}")

    from huggingface_hub import snapshot_download  # lazy load

    try:
        # Download snapshot from Hugging Face Hub
        snapshot_download(
            repo_id=hf_id,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            max_workers=4,
            token=settings.hf_token,
        )
        ctx.log(f"Download complete: {hf_id}")
    except Exception as e:
        ctx.log(f"Download failed: {e}")
        raise e

    # Calculate download size
    size_bytes = 0
    file_count = 0
    for f in dest_dir.glob("**/*"):
        if f.is_file():
            size_bytes += f.stat().st_size
            file_count += 1

    ctx.log(f"Downloaded {file_count} files ({size_bytes} bytes). Registering model...")

    # Create slug: lowercase, clean special chars
    slug = safe_name.lower()
    slug = re.sub(r"[^a-z0-9\-_.]", "-", slug)

    model_id = None
    with session_scope() as s:
        # Check if the model already exists in this project
        existing = (
            s.query(Model)
            .filter(Model.project_id == project_id, Model.slug == slug)
            .first()
        )
        if existing:
            model_id = existing.id
            ctx.log(f"Model already registered: {slug} (id={model_id})")
        else:
            from oas_core.registry import create_model
            model_id = create_model(
                project_id=project_id,
                slug=slug,
                name=hf_id,
                modality=modality,
                family=family,
                description=f"Imported from Hugging Face: {hf_id}",
            )
            ctx.log(f"Registered new Model: {slug} (id={model_id})")

        # Now register this version
        from oas_core.db import ModelVersion as ORMModelVersion
        from oas_core.registry import publish_version

        existing_ver = (
            s.query(ORMModelVersion)
            .filter(ORMModelVersion.model_id == model_id, ORMModelVersion.version == version)
            .first()
        )
        if existing_ver:
            ctx.log(f"Model version {version} already exists. Updating URI and size...")
            existing_ver.artifact_uri = f"file://{dest_dir}"
            existing_ver.size_bytes = size_bytes
            existing_ver.config = {"hf_model_id": hf_id}
        else:
            publish_version(
                model_id=model_id,
                version=version,
                artifact_uri=f"file://{dest_dir}",
                format="safetensors",
                size_bytes=size_bytes,
                config={"hf_model_id": hf_id},
                notes=f"Successfully downloaded from Hugging Face: {hf_id}",
                stage=ModelStage.DEV,
            )
            ctx.log(f"Published model version {version}")

    return {
        "hf_id": hf_id,
        "local_dir": str(dest_dir),
        "size_bytes": size_bytes,
        "model_id": model_id,
        "version": version,
    }


with contextlib.suppress(ValueError):
    register_handler("hf_model_download", hf_model_download_handler)
