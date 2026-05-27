"""Plan-and-PR generator.

The studio knows enough about handlers, datasets, and base models to write a
*self-describing* fine-tune plan: a JSON config that can be submitted as-is to
the queue, plus a markdown PR body summarising what changes and why.

Endpoints:
  POST /plans/finetune                  — generate a plan (no side effects)
  POST /plans/finetune/{plan_id}/submit — submit it to the queue
  POST /plans/finetune/{plan_id}/pr     — open a PR via `gh` if available
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from oas_core.db import Dataset, DatasetVersion, Project, Role, session_scope
from oas_core.queue import submit_job
from oas_core.settings import get_settings
from pydantic import BaseModel, Field

from oas_server.auth import CurrentUser, assert_role, require_user

router = APIRouter(prefix="/plans", tags=["plans"])


# In-memory plan store keyed by short id. Plans are ephemeral by design — they
# live until the user submits / discards them. Persisting them would invite
# stale configs.
_PLANS: dict[str, dict[str, Any]] = {}


class PlanIn(BaseModel):
    project_id: str
    modality: Literal["asr", "llm", "tts"]
    dataset_version_id: str
    base_model: str
    intent: str = Field(
        default="",
        description="One-sentence description of what this fine-tune is for. Shapes the PR body.",
    )
    mode: Literal["lora", "full"] = "lora"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    publish_model_slug: str | None = None
    publish_version: str = "0.1.0"


class PlanOut(BaseModel):
    plan_id: str
    job_kind: str
    job_name: str
    config: dict[str, Any]
    pr_title: str
    pr_body: str
    created_at: datetime


def _job_kind_for(modality: str) -> str:
    return {
        "asr": "whisper_finetune",
        "llm": "llm_finetune_sft",
        "tts": "tts_finetune_piper",
    }[modality]


def _build_config(body: PlanIn) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dataset_version_id": body.dataset_version_id,
        "base_model": body.base_model,
    }
    if body.modality == "asr":
        base["training"] = {
            "mode": body.mode,
            "epochs": body.epochs,
            "batch_size": body.batch_size,
            "learning_rate": body.learning_rate,
            "max_audio_s": 30,
            "language": "en",
            "task": "transcribe",
            "fp16": True,
        }
        base["lora"] = {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        }
    elif body.modality == "llm":
        base["training"] = {
            "epochs": body.epochs,
            "batch_size": body.batch_size,
            "grad_accum_steps": 4,
            "learning_rate": body.learning_rate,
            "max_seq_len": 2048,
            "bf16": True,
            "gradient_checkpointing": True,
        }
        base["lora"] = {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "auto",
        }
        base["quantization"] = "none"
    elif body.modality == "tts":
        base.pop("base_model")
        base["voice_name"] = body.publish_model_slug or "voice"
        base["language"] = "en"
        base["sample_rate"] = 22050
        base["training"] = {
            "max_epochs": max(body.epochs * 100, 200),
            "batch_size": body.batch_size,
            "checkpoint_epochs": 50,
            "quality": "medium",
        }
    return base


def _pr_body(body: PlanIn, dataset_summary: dict[str, Any], config: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"## Fine-tune plan: {body.modality.upper()} from `{body.base_model}`",
            "",
            body.intent or "_No intent description provided._",
            "",
            "### Data",
            f"- Dataset version: `{body.dataset_version_id}`",
            f"- Samples: **{dataset_summary.get('num_samples', '?')}**"
            f" · audio: {dataset_summary.get('total_audio_s', 0):.1f}s",
            f"- Source: `{dataset_summary.get('source') or 'manifest'}`",
            "",
            "### Training",
            f"- Mode: **{body.mode}** · epochs: **{body.epochs}**"
            f" · batch: **{body.batch_size}** · lr: **{body.learning_rate}**",
            "",
            "### Promotion",
            (
                f"- Will publish a `ModelVersion` `{body.publish_version}` under model "
                f"slug `{body.publish_model_slug}` on success."
                if body.publish_model_slug
                else "- Will _not_ publish a ModelVersion (no slug provided)."
            ),
            "",
            "### Reproduce",
            "```json",
            json.dumps(config, indent=2),
            "```",
            "",
            "_Generated by Open Audio Studio plans._",
        ]
    )


@router.post("/finetune", response_model=PlanOut)
def plan_finetune(body: PlanIn, user: CurrentUser = Depends(require_user)) -> PlanOut:
    assert_role(user, body.project_id, Role.EDITOR)

    with session_scope() as s:
        if not s.get(Project, body.project_id):
            raise HTTPException(404, "project not found")
        dv = s.get(DatasetVersion, body.dataset_version_id)
        if not dv:
            raise HTTPException(404, "dataset version not found")
        ds = s.get(Dataset, dv.dataset_id)
        if ds and ds.modality.value != body.modality:
            raise HTTPException(
                400, f"dataset modality {ds.modality.value!r} != requested {body.modality!r}"
            )
        dataset_summary = {
            "num_samples": dv.num_samples,
            "total_audio_s": dv.total_audio_s,
            "source": ds.source if ds else None,
        }

    config = _build_config(body)
    if body.publish_model_slug:
        config["registry"] = {
            "model_id": None,  # resolved at submit time
            "version": body.publish_version,
            "slug": body.publish_model_slug,
        }
    pr_title = f"finetune({body.modality}): {body.base_model.split('/')[-1]}"
    pr_body = _pr_body(body, dataset_summary, config)

    plan_id = secrets.token_urlsafe(6)
    plan = {
        "plan_id": plan_id,
        "owner": user.id,
        "project_id": body.project_id,
        "job_kind": _job_kind_for(body.modality),
        "job_name": f"{body.modality}-ft {body.base_model.split('/')[-1]}",
        "config": config,
        "pr_title": pr_title,
        "pr_body": pr_body,
        "created_at": datetime.now(UTC),
    }
    _PLANS[plan_id] = plan
    # Use model_validate to avoid `dict[str, object]` invariance under **unpack;
    # pydantic coerces fields per the schema declarations on PlanOut.
    return PlanOut.model_validate({k: plan[k] for k in PlanOut.model_fields})


class SubmitOut(BaseModel):
    job_id: str


@router.post("/finetune/{plan_id}/submit", response_model=SubmitOut)
def submit_plan(plan_id: str, user: CurrentUser = Depends(require_user)) -> SubmitOut:
    plan = _PLANS.get(plan_id)
    if not plan:
        raise HTTPException(404, "plan not found or expired")
    assert_role(user, plan["project_id"], Role.EDITOR)

    config = dict(plan["config"])
    reg = config.pop("registry", None)
    if reg and reg.get("slug"):
        # Materialize the Model row on demand if it doesn't exist.
        from oas_core.db import Model
        from oas_core.db.models import Modality

        with session_scope() as s:
            modality = (
                Modality.ASR
                if plan["job_kind"].startswith("whisper")
                else Modality.LLM
                if plan["job_kind"].startswith("llm")
                else Modality.TTS
            )
            existing = s.query(Model).filter(
                Model.project_id == plan["project_id"], Model.slug == reg["slug"]
            ).first()
            if existing:
                model_id = existing.id
            else:
                m = Model(
                    project_id=plan["project_id"],
                    slug=reg["slug"],
                    name=reg["slug"],
                    modality=modality,
                )
                s.add(m)
                s.flush()
                model_id = m.id
        config["registry"] = {"model_id": model_id, "version": reg["version"]}

    job_id = submit_job(plan["project_id"], plan["job_kind"], plan["job_name"], config)
    return SubmitOut(job_id=job_id)


class PROut(BaseModel):
    pr_url: str | None
    branch: str
    method: str  # 'gh-cli' | 'patch-only'
    patch_path: str | None = None


class PRIn(BaseModel):
    repo: str | None = Field(default=None, description="<owner>/<name> (uses cwd repo if omitted)")
    base: str = "main"
    branch: str | None = None


@router.post("/finetune/{plan_id}/pr", response_model=PROut)
def open_pr(plan_id: str, body: PRIn, user: CurrentUser = Depends(require_user)) -> PROut:
    plan = _PLANS.get(plan_id)
    if not plan:
        raise HTTPException(404, "plan not found or expired")
    assert_role(user, plan["project_id"], Role.ADMIN)

    settings = get_settings()
    plans_dir = settings.data_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    config_path = plans_dir / f"{plan_id}.json"
    config_path.write_text(json.dumps(plan["config"], indent=2))
    body_path = plans_dir / f"{plan_id}.md"
    body_path.write_text(plan["pr_body"])

    # Try the gh CLI; otherwise return the artifact paths so the user can open
    # the PR manually.
    import shutil

    if shutil.which("gh") is None:
        return PROut(pr_url=None, branch=body.branch or f"oas/plan/{plan_id}", method="patch-only", patch_path=str(config_path))

    branch = body.branch or f"oas/plan/{plan_id}"
    try:
        with tempfile.TemporaryDirectory() as _td:
            # The simplest path: assume the user is running the studio inside
            # a git checkout. We commit + push from that working tree.
            env = os.environ.copy()
            subprocess.check_call(["git", "checkout", "-b", branch], cwd=os.getcwd(), env=env)
            target = Path(os.getcwd()) / "configs" / "plans"
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{plan_id}.json").write_text(config_path.read_text())
            subprocess.check_call(["git", "add", str(target)], cwd=os.getcwd(), env=env)
            subprocess.check_call(["git", "commit", "-m", plan["pr_title"]], cwd=os.getcwd(), env=env)
            subprocess.check_call(["git", "push", "-u", "origin", branch], cwd=os.getcwd(), env=env)
            cmd = ["gh", "pr", "create", "--title", plan["pr_title"], "--body-file", str(body_path), "--base", body.base]
            if body.repo:
                cmd += ["--repo", body.repo]
            url = subprocess.check_output(cmd, cwd=os.getcwd(), env=env, text=True).strip()
        return PROut(pr_url=url.splitlines()[-1], branch=branch, method="gh-cli")
    except Exception as e:
        raise HTTPException(500, f"failed to open PR: {e}") from e
