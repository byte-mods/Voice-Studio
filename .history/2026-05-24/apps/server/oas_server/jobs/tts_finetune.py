"""Job handler: fine-tune a Piper TTS voice from a TTS manifest.

Piper's training pipeline uses VITS under the hood. This handler:
  1. Reads TTSSample entries from a dataset version.
  2. Resamples audio to Piper's target rate (22.05 kHz mono).
  3. Builds Piper's expected metadata.csv / wavs/ layout in a working dir.
  4. Runs preprocessing + training via `python -m piper_train` subprocess (Piper
     keeps its trainer as a CLI; we shell out so we don't depend on its
     internal Python API).
  5. Exports the resulting .onnx + .json voice, publishes a ModelVersion.

Config:

    {
      "dataset_version_id": "...",
      "voice_name": "my-voice",
      "language": "en",
      "sample_rate": 22050,                 # Piper supports 16k or 22.05k
      "training": {
        "max_epochs": 1000,
        "batch_size": 32,
        "checkpoint_epochs": 50,
        "quality": "medium"                  # 'x_low' | 'low' | 'medium' | 'high'
      },
      "base_voice": "en_US-lessac-medium",  # optional warm-start
      "registry": {"model_id": "...", "version": "0.1.0"}
    }

Heavy deps (`torch`, `piper-train`, `soundfile`, `librosa`) are imported
lazily and installed via `pip install -e 'apps/server[tts]'`.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from oas_core.db import DatasetVersion, session_scope
from oas_core.manifest import ManifestReader, Modality, TTSSample
from oas_core.queue.backend import JobContext, register_handler
from oas_core.registry import publish_version

log = logging.getLogger(__name__)


def _load_tts_split(manifest_root: Path, split: str = "train") -> list[TTSSample]:
    out: list[TTSSample] = []
    for s in ManifestReader(manifest_root):
        if s.modality != Modality.TTS:
            continue
        if s.split.value != split:
            continue
        out.append(s)
    return out


def _prepare_piper_dataset(
    samples: list[TTSSample], out_dir: Path, target_sr: int, ctx: JobContext
) -> Path:
    """Lay out samples as Piper expects: wavs/<id>.wav + metadata.csv."""
    import soundfile as sf

    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"

    with meta_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, delimiter="|")
        for s in samples:
            src_path = Path(s.audio.uri.removeprefix("file://"))
            try:
                data, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
            except Exception as e:
                ctx.log(f"skip {s.id}: read failed: {e}")
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != target_sr:
                try:
                    import librosa

                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                except ImportError as e:
                    raise RuntimeError("install librosa to resample TTS audio") from e
            dst = wavs / f"{s.id}.wav"
            sf.write(str(dst), data, target_sr, subtype="PCM_16")
            writer.writerow([s.id, s.text, s.speaker_id])

    return meta_path


def tts_finetune_handler(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    dv_id: str = cfg["dataset_version_id"]
    voice_name: str = cfg.get("voice_name", "voice")
    language: str = cfg.get("language", "en")
    sample_rate: int = int(cfg.get("sample_rate", 22050))
    training_cfg = cfg.get("training", {})
    base_voice: str | None = cfg.get("base_voice")
    registry_cfg = cfg.get("registry") or {}

    max_epochs = int(training_cfg.get("max_epochs", 1000))
    batch_size = int(training_cfg.get("batch_size", 32))
    checkpoint_epochs = int(training_cfg.get("checkpoint_epochs", 50))
    quality = str(training_cfg.get("quality", "medium"))

    with session_scope() as s:
        v = s.get(DatasetVersion, dv_id)
        if not v:
            raise ValueError(f"DatasetVersion {dv_id!r} not found")
        manifest_root = Path(v.manifest_uri.removeprefix("file://"))

    ctx.log(f"loading TTS samples from {manifest_root}")
    samples = _load_tts_split(manifest_root, "train")
    ctx.log(f"loaded {len(samples)} TTS train samples")
    if not samples:
        raise ValueError("no TTS train samples in manifest")

    work = Path(ctx.artifacts_dir) / "piper"
    work.mkdir(parents=True, exist_ok=True)
    dataset_dir = work / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    ctx.log(f"materializing dataset at {dataset_dir}")
    _prepare_piper_dataset(samples, dataset_dir, sample_rate, ctx)

    # ---- Preprocess: piper_train.preprocess ----
    cache_dir = work / "cache"
    cache_dir.mkdir(exist_ok=True)
    pre_cmd = [
        sys.executable,
        "-m",
        "piper_train.preprocess",
        "--language", language,
        "--input-dir", str(dataset_dir),
        "--output-dir", str(cache_dir),
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", str(sample_rate),
    ]
    ctx.log(f"$ {' '.join(pre_cmd)}")
    _run_streaming(pre_cmd, ctx)

    # ---- Train: piper_train ----
    ckpt_dir = work / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    train_cmd = [
        sys.executable,
        "-m",
        "piper_train",
        "--dataset-dir", str(cache_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", str(batch_size),
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(max_epochs),
        "--checkpoint-epochs", str(checkpoint_epochs),
        "--precision", "32",
        "--quality", quality,
        "--default_root_dir", str(ckpt_dir),
    ]
    if base_voice:
        train_cmd += ["--resume_from_checkpoint", base_voice]
    ctx.log(f"$ {' '.join(train_cmd)}")
    _run_streaming(train_cmd, ctx)

    # ---- Export: piper_train.export_onnx ----
    last_ckpt = _find_latest_ckpt(ckpt_dir)
    if not last_ckpt:
        raise RuntimeError("no checkpoint produced by piper_train")

    out_voice_dir = Path(ctx.artifacts_dir) / "voice"
    out_voice_dir.mkdir(exist_ok=True)
    onnx_path = out_voice_dir / f"{voice_name}.onnx"
    json_path = out_voice_dir / f"{voice_name}.onnx.json"

    export_cmd = [
        sys.executable,
        "-m",
        "piper_train.export_onnx",
        str(last_ckpt),
        str(onnx_path),
    ]
    ctx.log(f"$ {' '.join(export_cmd)}")
    _run_streaming(export_cmd, ctx)

    # Copy the config json that piper_train wrote next to the checkpoint.
    cfg_src = last_ckpt.parent / "config.json"
    if cfg_src.exists():
        shutil.copy(cfg_src, json_path)

    artifact_uri = f"file://{out_voice_dir}"
    ctx.log(f"voice exported to {artifact_uri}")

    if registry_cfg.get("model_id") and registry_cfg.get("version"):
        try:
            publish_version(
                model_id=registry_cfg["model_id"],
                version=registry_cfg["version"],
                artifact_uri=artifact_uri,
                format="piper-onnx",
                source_run_id=ctx.run_id,
                source_dataset_version_id=dv_id,
                notes=f"Piper voice {voice_name} ({quality})",
            )
            ctx.log(f"published voice {registry_cfg['version']}")
        except Exception as e:
            ctx.log(f"publish failed: {e}")

    return {
        "voice_name": voice_name,
        "samples": len(samples),
        "artifact_uri": artifact_uri,
        "checkpoint": str(last_ckpt),
    }


def _run_streaming(cmd: list[str], ctx: JobContext) -> None:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        ctx.log(line.rstrip())
        if ctx.cancelled:
            proc.terminate()
            break
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"subprocess exited {rc}: {' '.join(cmd[:3])}")


def _find_latest_ckpt(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


with contextlib.suppress(ValueError):
    register_handler("tts_finetune_piper", tts_finetune_handler)
