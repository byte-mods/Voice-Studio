"""Job handler: SFT-fine-tune a native audio-LM on S2S manifest samples.

Targets multimodal chat models (Qwen2.5-Omni / LLaMA-Omni / GLM-4-Voice) where
each turn can carry text or audio. We materialize each S2S sample as a multi-
turn chat where user turns contain audio and assistant turns contain text
(audio-out training is gated behind a per-model flag and not yet enabled here;
audio output finetuning typically requires the codec head's own loss term).

Config:

    {
      "dataset_version_id": "...",
      "base_model": "Qwen/Qwen2.5-Omni-7B",
      "training": {
        "epochs": 1,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "learning_rate": 1e-4,
        "max_audio_s": 30,
        "max_seq_len": 4096,
        "bf16": true,
        "gradient_checkpointing": true
      },
      "lora": {
        "r": 16, "alpha": 32, "dropout": 0.05,
        "target_modules": "auto"
      },
      "registry": {"model_id": "...", "version": "0.1.0"}
    }

Heavy deps are imported lazily; install via `pip install -e 'apps/server[s2s]'`.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from oas_core.db import DatasetVersion, session_scope
from oas_core.manifest import ManifestReader, Modality, S2SSample
from oas_core.queue.backend import JobContext, register_handler
from oas_core.registry import publish_version

log = logging.getLogger(__name__)


def _load_s2s_split(manifest_root: Path, split: str) -> list[S2SSample]:
    out: list[S2SSample] = []
    for s in ManifestReader(manifest_root):
        if s.modality != Modality.S2S:
            continue
        if s.split.value != split:
            continue
        out.append(s)
    return out


def _load_audio(uri: str, target_sr: int = 16000) -> Any:
    import numpy as np
    import soundfile as sf

    path = Path(uri.removeprefix("file://"))
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError as e:
            raise RuntimeError(f"audio is {sr} Hz; install librosa to resample") from e
    return np.asarray(data, dtype=np.float32), target_sr


def _build_chat(sample: S2SSample, max_audio_s: float) -> list[dict[str, Any]]:
    """Render an S2SSample as a chat-template-compatible message list."""
    messages: list[dict[str, Any]] = []
    for turn in sample.turns:
        content: list[dict[str, Any]] = []
        if turn.audio is not None:
            audio, sr = _load_audio(turn.audio.uri)
            if max_audio_s and len(audio) > int(max_audio_s * sr):
                audio = audio[: int(max_audio_s * sr)]
            content.append({"type": "audio", "audio": audio, "sampling_rate": sr})
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        if not content:
            continue
        messages.append({"role": turn.role.value, "content": content})
    return messages


def _find_linear_module_names(model: Any) -> list[str]:
    import torch.nn as nn

    types: tuple[type, ...] = (nn.Linear,)
    try:
        import bitsandbytes as bnb

        types = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        pass
    names: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, types):
            short = name.split(".")[-1]
            if short in ("lm_head", "audio_decoder"):
                continue
            names.add(short)
    return sorted(names)


def s2s_native_finetune_handler(ctx: JobContext) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoModel,
        AutoProcessor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    cfg = ctx.config
    dv_id = cfg["dataset_version_id"]
    base_model = cfg["base_model"]
    training_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    registry_cfg = cfg.get("registry") or {}

    epochs = int(training_cfg.get("epochs", 1))
    batch_size = int(training_cfg.get("batch_size", 1))
    grad_accum = int(training_cfg.get("grad_accum_steps", 8))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    max_audio_s = float(training_cfg.get("max_audio_s", 30))
    max_seq_len = int(training_cfg.get("max_seq_len", 4096))
    bf16 = bool(training_cfg.get("bf16", True))
    grad_ckpt = bool(training_cfg.get("gradient_checkpointing", True))

    with session_scope() as s:
        v = s.get(DatasetVersion, dv_id)
        if not v:
            raise ValueError(f"DatasetVersion {dv_id!r} not found")
        manifest_root = Path(v.manifest_uri.removeprefix("file://"))

    ctx.log(f"loading S2S samples from {manifest_root}")
    train_samples = _load_s2s_split(manifest_root, "train")
    eval_samples = _load_s2s_split(manifest_root, "val")
    ctx.log(f"train={len(train_samples)}  eval={len(eval_samples)}")
    if not train_samples:
        raise ValueError("no S2S train samples in manifest")

    ctx.log(f"loading multimodal model+processor: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True)
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    target_modules = lora_cfg.get("target_modules", "auto")
    if target_modules == "auto":
        target_modules = _find_linear_module_names(model)
        ctx.log(f"auto LoRA target modules ({len(target_modules)}): {target_modules}")
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    class _DS(TorchDataset):  # type: ignore[misc]
        def __init__(self, samples: list[S2SSample]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            sample = self.samples[idx]
            messages = _build_chat(sample, max_audio_s)
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    return_tensors="pt",
                    tokenize=True,
                    truncation=True,
                    max_length=max_seq_len,
                )
                if not isinstance(inputs, dict):
                    inputs = {"input_ids": inputs}
            except Exception:
                # Fallback: stringify the turns and let the processor tokenize.
                text = "\n".join(
                    f"{m['role']}: "
                    + " ".join(
                        c.get("text", "[audio]") for c in m["content"] if isinstance(c, dict)
                    )
                    for m in messages
                )
                inputs = processor(text=text, return_tensors="pt", truncation=True, max_length=max_seq_len)

            labels = inputs["input_ids"].clone() if "input_ids" in inputs else None
            out: dict[str, Any] = {k: v.squeeze(0) for k, v in inputs.items() if hasattr(v, "squeeze")}
            if labels is not None:
                out["labels"] = labels.squeeze(0)
            return out

    train_ds = _DS(train_samples)
    eval_ds = _DS(eval_samples) if eval_samples else None

    out_dir = Path(ctx.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=bf16 and torch.cuda.is_available(),
        gradient_checkpointing=grad_ckpt,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    class _Cb(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            if logs:
                ctx.log(
                    "step "
                    + " ".join(f"{k}={v}" for k, v in logs.items() if isinstance(v, (int, float)))
                )
                ctx.heartbeat()

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Any = None, **_: Any) -> None:
            ctx.log(f"eval: {metrics}")

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Padding shape can vary per model; defer to the processor where possible.
        return processor.tokenizer.pad(batch, return_tensors="pt") if hasattr(processor, "tokenizer") else batch[0]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collate,
        callbacks=[_Cb()],
    )

    ctx.log("starting native-S2S SFT")
    trainer.train()
    ctx.log("training done")

    final_metrics: dict[str, Any] = {}
    if eval_ds is not None:
        final_metrics = {
            k: float(v)
            for k, v in trainer.evaluate().items()
            if isinstance(v, (int, float))
        }

    artifact_path = out_dir / "final"
    trainer.save_model(str(artifact_path))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(artifact_path))
    ctx.log(f"saved adapter to {artifact_path}")

    if registry_cfg.get("model_id") and registry_cfg.get("version"):
        try:
            publish_version(
                model_id=registry_cfg["model_id"],
                version=registry_cfg["version"],
                artifact_uri=f"file://{artifact_path}",
                format="peft-adapter-omni",
                metrics=final_metrics,
                source_run_id=ctx.run_id,
                source_dataset_version_id=dv_id,
                notes=f"Native S2S SFT from {base_model}",
            )
            ctx.log(f"published native S2S adapter {registry_cfg['version']}")
        except Exception as e:
            ctx.log(f"publish failed: {e}")

    return {
        "base_model": base_model,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "artifact_uri": f"file://{artifact_path}",
        **final_metrics,
    }


with contextlib.suppress(ValueError):
    register_handler("s2s_native_finetune", s2s_native_finetune_handler)
