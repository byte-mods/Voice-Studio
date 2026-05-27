"""Job handler: evaluate a fine-tuned LLM on the test split of an LLM dataset.

Reports:
- mean cross-entropy loss on the held-out conversations
- per-token perplexity
- response-length distribution

Config:

    {
      "model_version_id": "...",       # PEFT adapter or full model
      "dataset_version_id": "...",
      "split": "test",
      "max_samples": 200,
      "base_model": "Qwen/Qwen2.5-0.5B-Instruct"  # required for adapter loading
    }
"""

from __future__ import annotations

import contextlib
import logging
import math
from pathlib import Path
from typing import Any

from oas_core.db import DatasetVersion, ModelVersion, session_scope
from oas_core.manifest import LLMSample, ManifestReader, Modality
from oas_core.queue.backend import JobContext, register_handler

log = logging.getLogger(__name__)


def _load_llm_split(manifest_root: Path, split: str) -> list[LLMSample]:
    out: list[LLMSample] = []
    for s in ManifestReader(manifest_root):
        if s.modality != Modality.LLM:
            continue
        if s.split.value != split:
            continue
        out.append(s)
    return out


def llm_eval_handler(ctx: JobContext) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = ctx.config
    mv_id = cfg["model_version_id"]
    dv_id = cfg["dataset_version_id"]
    split = cfg.get("split", "test")
    max_samples = cfg.get("max_samples", 200)
    base_model = cfg.get("base_model")

    with session_scope() as db:
        mv = db.get(ModelVersion, mv_id)
        dv = db.get(DatasetVersion, dv_id)
        if not mv or not dv:
            raise ValueError("model_version_id or dataset_version_id not found")
        adapter_path = Path(mv.artifact_uri.removeprefix("file://"))
        manifest_root = Path(dv.manifest_uri.removeprefix("file://"))
        fmt = mv.format

    samples = _load_llm_split(manifest_root, split)[: int(max_samples)]
    if not samples:
        raise ValueError(f"no LLM samples in split {split!r}")
    ctx.log(f"evaluating on {len(samples)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if fmt == "peft-adapter":
        if not base_model:
            raise ValueError("base_model required to evaluate a peft-adapter")
        from peft import PeftModel

        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype).to(device)
        model = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(str(adapter_path), use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(str(adapter_path), torch_dtype=dtype).to(device)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    asst_lengths: list[int] = []

    for i, s in enumerate(samples):
        if ctx.cancelled:
            break
        messages = []
        if s.system_prompt:
            messages.append({"role": "system", "content": s.system_prompt})
        for turn in s.turns:
            messages.append({"role": turn.role.value, "content": turn.text or ""})
            if turn.role.value == "assistant" and turn.text:
                asst_lengths.append(len(tok.encode(turn.text)))
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n_tok = int(enc["input_ids"].numel())
        total_loss += float(out.loss.item()) * n_tok
        total_tokens += n_tok
        if i % 10 == 0:
            ctx.log(f"  step processed={i} mean_loss={total_loss / max(total_tokens, 1):.4f}")
            ctx.heartbeat()

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss) if mean_loss < 30 else float("inf")
    asst_lengths.sort()
    length_summary = {
        "min": asst_lengths[0] if asst_lengths else 0,
        "median": asst_lengths[len(asst_lengths) // 2] if asst_lengths else 0,
        "p95": asst_lengths[int(len(asst_lengths) * 0.95)] if asst_lengths else 0,
        "max": asst_lengths[-1] if asst_lengths else 0,
        "n": len(asst_lengths),
    }
    ctx.log(f"step eval_loss={mean_loss:.4f} perplexity={ppl:.4f}")
    return {
        "eval_loss": mean_loss,
        "perplexity": ppl,
        "n_samples": len(samples),
        "n_tokens": total_tokens,
        "assistant_token_lengths": length_summary,
    }


with contextlib.suppress(ValueError):
    register_handler("llm_eval", llm_eval_handler)
