"""Job handler: LoRA SFT fine-tune for a chat LLM.

Config:

    {
      "dataset_version_id": "...",
      "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
      "training": {
        "epochs": 3,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "max_seq_len": 2048,
        "bf16": true,
        "gradient_checkpointing": true
      },
      "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "auto"      # 'auto' picks all linear layers
      },
      "quantization": "none",          # 'none' | '4bit' | '8bit'
      "packing": false,
      "registry": {"model_id": "...", "version": "0.1.0"}
    }

Reads LLM samples from the manifest, renders each via the base model's chat
template, packs into a HF Dataset, and trains with `trl.SFTTrainer`. Lazy-imports
all heavy deps.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from oas_core.db import DatasetVersion, session_scope
from oas_core.manifest import LLMSample, ManifestReader, Modality
from oas_core.queue.backend import JobContext, register_handler
from oas_core.registry import publish_version

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


def _render_chat(sample: LLMSample, tokenizer: Any) -> str:
    messages: list[dict[str, str]] = []
    if sample.system_prompt:
        messages.append({"role": "system", "content": sample.system_prompt})
    for turn in sample.turns:
        messages.append({"role": turn.role.value, "content": turn.text or ""})
    try:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    except Exception:
        # Fall back to a simple concatenation if the tokenizer has no template.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def llm_finetune_handler(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    dv_id: str = cfg["dataset_version_id"]
    base_model: str = cfg.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
    training_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    quantization = cfg.get("quantization", "none")
    packing = bool(cfg.get("packing", False))
    registry_cfg = cfg.get("registry") or {}

    epochs = int(training_cfg.get("epochs", 3))
    batch_size = int(training_cfg.get("batch_size", 4))
    grad_accum = int(training_cfg.get("grad_accum_steps", 4))
    lr = float(training_cfg.get("learning_rate", 2e-4))
    warmup = float(training_cfg.get("warmup_ratio", 0.03))
    max_seq_len = int(training_cfg.get("max_seq_len", 2048))
    bf16 = bool(training_cfg.get("bf16", True))
    grad_ckpt = bool(training_cfg.get("gradient_checkpointing", True))

    with session_scope() as s:
        v = s.get(DatasetVersion, dv_id)
        if not v:
            raise ValueError(f"DatasetVersion {dv_id!r} not found")
        manifest_root = Path(v.manifest_uri.removeprefix("file://"))

    ctx.log(f"loading LLM samples from {manifest_root}")
    train_samples = _load_llm_split(manifest_root, "train")
    eval_samples = _load_llm_split(manifest_root, "val")
    ctx.log(f"train={len(train_samples)}  eval={len(eval_samples)}")
    if not train_samples:
        raise ValueError("no LLM train samples in manifest")

    # ---- heavy imports ----
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    ctx.log(f"loading tokenizer + model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16 if bf16 else torch.float32}
    if quantization in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    target_modules = lora_cfg.get("target_modules", "auto")
    if target_modules == "auto":
        target_modules = _find_linear_module_names(model)
        ctx.log(f"auto-picked LoRA target modules: {target_modules}")

    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    def _to_ds(samples: list[LLMSample]) -> HFDataset:
        return HFDataset.from_list([{"text": _render_chat(s, tokenizer)} for s in samples])

    train_ds = _to_ds(train_samples)
    eval_ds = _to_ds(eval_samples) if eval_samples else None

    out_dir = Path(ctx.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup,
        bf16=bf16 and torch.cuda.is_available(),
        gradient_checkpointing=grad_ckpt,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=2,
        max_seq_length=max_seq_len,
        packing=packing,
        dataset_text_field="text",
        report_to=[],
    )

    class _LogCallback(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            if logs:
                ctx.log("step " + " ".join(f"{k}={v}" for k, v in logs.items() if isinstance(v, (int, float))))
                ctx.heartbeat()

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Any = None, **_: Any) -> None:
            ctx.log(f"eval: {metrics}")

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_cfg,
        tokenizer=tokenizer,
        callbacks=[_LogCallback()],
    )

    ctx.log("starting training")
    trainer.train()
    ctx.log("training done")

    final_metrics: dict[str, Any] = {}
    if eval_ds is not None:
        final_metrics = {
            k: float(v) for k, v in trainer.evaluate().items() if isinstance(v, (int, float))
        }

    artifact_path = out_dir / "final"
    trainer.save_model(str(artifact_path))
    tokenizer.save_pretrained(str(artifact_path))
    ctx.log(f"saved adapter to {artifact_path}")

    if registry_cfg.get("model_id") and registry_cfg.get("version"):
        try:
            publish_version(
                model_id=registry_cfg["model_id"],
                version=registry_cfg["version"],
                artifact_uri=f"file://{artifact_path}",
                format="peft-adapter",
                metrics=final_metrics,
                source_run_id=ctx.run_id,
                source_dataset_version_id=dv_id,
                notes=f"LoRA SFT from {base_model}",
            )
            ctx.log(f"published version {registry_cfg['version']}")
        except Exception as e:
            ctx.log(f"publish failed: {e}")

    return {
        "base_model": base_model,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "artifact_uri": f"file://{artifact_path}",
        **final_metrics,
    }


def _find_linear_module_names(model: Any) -> list[str]:
    """Discover all `nn.Linear` (and bnb Linear4bit / Linear8bit) module
    short-names. PEFT target_modules wants name suffixes, not full dotted paths."""
    import torch.nn as nn

    linear_types: tuple[type, ...] = (nn.Linear,)
    try:
        import bitsandbytes as bnb

        linear_types = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        pass

    names: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, linear_types):
            # Skip the LM head (a common foot-gun).
            short = name.split(".")[-1]
            if short == "lm_head":
                continue
            names.add(short)
    return sorted(names)


with contextlib.suppress(ValueError):
    register_handler("llm_finetune_sft", llm_finetune_handler)
