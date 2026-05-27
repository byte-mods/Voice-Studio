"""Job handler: fine-tune a Whisper-family ASR model with LoRA.

Config:

    {
      "dataset_version_id": "...",          # required
      "base_model": "openai/whisper-small", # default
      "training": {
        "mode": "lora",                     # 'lora' | 'full'
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.1,
        "max_audio_s": 30,
        "language": "en",
        "task": "transcribe",
        "fp16": true
      },
      "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
      },
      "registry": {                          # optional auto-publish
        "model_id": "...",                   # OAS Model.id; if missing, no publish
        "version": "0.1.0"
      },
      "eval_split": "val"
    }

Heavy deps (`torch`, `transformers`, `peft`, `evaluate`, `accelerate`,
`datasets`, `soundfile`) are imported lazily so the studio still boots
without them. Install via `pip install -e 'apps/server[asr]'`.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from oas_core.db import DatasetVersion, session_scope
from oas_core.manifest import ASRSample, ManifestReader, Modality
from oas_core.queue.backend import JobContext, register_handler
from oas_core.registry import publish_version

log = logging.getLogger(__name__)


def _load_manifest_split(manifest_root: Path, split: str) -> list[ASRSample]:
    out: list[ASRSample] = []
    for s in ManifestReader(manifest_root):
        if s.modality != Modality.ASR:
            continue
        if s.split.value != split:
            continue
        out.append(s)
    return out


def _make_dataset(samples: list[ASRSample], processor: Any, max_audio_s: float) -> Any:
    import soundfile as sf
    from torch.utils.data import Dataset

    class _DS(Dataset):  # type: ignore[misc]
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            s = samples[idx]
            path = Path(s.audio.uri.removeprefix("file://"))
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                # Resample on the fly. For best speed precompute outside the trainer.
                try:
                    import librosa

                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                except ImportError:
                    raise RuntimeError(f"sample {s.id} is {sr} Hz; install librosa to resample") from None
                sr = 16000
            if max_audio_s and len(data) > int(max_audio_s * sr):
                data = data[: int(max_audio_s * sr)]

            features = processor(data, sampling_rate=sr, return_tensors="pt").input_features[0]
            labels = processor.tokenizer(s.transcript or "", return_tensors="pt").input_ids[0]
            return {"input_features": features, "labels": labels}

    return _DS()


def _collator(processor: Any) -> Any:
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = torch.stack([b["input_features"] for b in batch])
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id or -100,
        )
        labels[labels == (processor.tokenizer.pad_token_id or -100)] = -100
        return {"input_features": input_features, "labels": labels}

    return collate


def whisper_finetune_handler(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    dv_id: str = cfg["dataset_version_id"]
    base_model: str = cfg.get("base_model", "openai/whisper-small")
    training_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    registry_cfg = cfg.get("registry") or {}

    mode = training_cfg.get("mode", "lora")
    epochs = int(training_cfg.get("epochs", 3))
    batch_size = int(training_cfg.get("batch_size", 8))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    max_audio_s = float(training_cfg.get("max_audio_s", 30))
    language = training_cfg.get("language", "en")
    task = training_cfg.get("task", "transcribe")
    fp16 = bool(training_cfg.get("fp16", True))

    out_dir = Path(ctx.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with session_scope() as s:
        v = s.get(DatasetVersion, dv_id)
        if not v:
            raise ValueError(f"DatasetVersion {dv_id!r} not found")
        manifest_root = Path(v.manifest_uri.removeprefix("file://"))

    ctx.log(f"loading samples from {manifest_root}")
    train_samples = _load_manifest_split(manifest_root, "train")
    eval_samples = _load_manifest_split(manifest_root, cfg.get("eval_split", "val"))
    ctx.log(f"train={len(train_samples)}  eval={len(eval_samples)}")
    if not train_samples:
        raise ValueError("no train samples in manifest")

    # ------------------------- heavy imports below -------------------------
    import evaluate
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    ctx.log(f"loading processor + model: {base_model}")
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            bias="none",
            task_type="SPEECH_SEQ_2_SEQ_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    train_ds = _make_dataset(train_samples, processor, max_audio_s)
    eval_ds = _make_dataset(eval_samples, processor, max_audio_s) if eval_samples else None
    collate = _collator(processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": float(wer) if wer is not None else 1.0}

    class _LogCallback(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            if logs:
                ctx.log(
                    "step "
                    + " ".join(f"{k}={v}" for k, v in logs.items() if isinstance(v, (int, float)))
                )
                ctx.heartbeat()

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Any = None, **_: Any) -> None:
            ctx.log(f"eval: {metrics}")

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        num_train_epochs=epochs,
        fp16=fp16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        callbacks=[_LogCallback()],
    )

    ctx.log("starting training")
    trainer.train()
    ctx.log("training done")

    final_metrics: dict[str, Any] = {}
    if eval_ds is not None:
        ctx.log("running final evaluation")
        final_metrics = {k: float(v) for k, v in trainer.evaluate().items() if isinstance(v, (int, float))}

    artifact_path = out_dir / "final"
    trainer.save_model(str(artifact_path))
    processor.save_pretrained(str(artifact_path))
    ctx.log(f"saved checkpoint to {artifact_path}")

    if registry_cfg.get("model_id") and registry_cfg.get("version"):
        try:
            publish_version(
                model_id=registry_cfg["model_id"],
                version=registry_cfg["version"],
                artifact_uri=f"file://{artifact_path}",
                format="hf",
                metrics=final_metrics,
                source_run_id=ctx.run_id,
                source_dataset_version_id=dv_id,
                notes=f"Fine-tuned from {base_model} ({mode})",
            )
            ctx.log(f"published model version {registry_cfg['version']}")
        except Exception as e:
            ctx.log(f"publish failed: {e}")

    return {
        "base_model": base_model,
        "mode": mode,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "artifact_uri": f"file://{artifact_path}",
        **final_metrics,
    }


with contextlib.suppress(ValueError):
    register_handler("whisper_finetune", whisper_finetune_handler)
