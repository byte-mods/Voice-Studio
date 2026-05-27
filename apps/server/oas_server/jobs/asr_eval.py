"""Job handler: evaluate an ASR ModelVersion on a dataset version's test split.

Config:

    {
      "model_version_id": "...",
      "dataset_version_id": "...",
      "split": "test",
      "language": "en",
      "max_samples": 500,
      "slice_by": ["language", "domain", "accent"]
    }

Returns sliced WER + sample-level error inventory in metrics.
"""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from oas_core.db import DatasetVersion, ModelVersion, session_scope
from oas_core.manifest import ASRSample, ManifestReader, Modality
from oas_core.queue.backend import JobContext, register_handler

log = logging.getLogger(__name__)


def _load_test(manifest_root: Path, split: str) -> list[ASRSample]:
    out: list[ASRSample] = []
    for s in ManifestReader(manifest_root):
        if s.modality != Modality.ASR:
            continue
        if s.split.value != split:
            continue
        out.append(s)
    return out


def asr_eval_handler(ctx: JobContext) -> dict[str, Any]:
    import jiwer
    import soundfile as sf

    cfg = ctx.config
    mv_id = cfg["model_version_id"]
    dv_id = cfg["dataset_version_id"]
    split = cfg.get("split", "test")
    language = cfg.get("language", "en")
    max_samples = cfg.get("max_samples")
    slice_by = list(cfg.get("slice_by") or ["language"])

    with session_scope() as db:
        mv = db.get(ModelVersion, mv_id)
        dv = db.get(DatasetVersion, dv_id)
        if not mv or not dv:
            raise ValueError("model_version_id or dataset_version_id not found")
        artifact_path = Path(mv.artifact_uri.removeprefix("file://"))
        manifest_root = Path(dv.manifest_uri.removeprefix("file://"))

    ctx.log(f"loading ASR model: {artifact_path}")
    asr = _load_asr(artifact_path, language)

    samples = _load_test(manifest_root, split)
    if max_samples:
        samples = samples[: int(max_samples)]
    ctx.log(f"evaluating on {len(samples)} samples (split={split!r})")

    refs: list[str] = []
    hyps: list[str] = []
    processed_samples: list[ASRSample] = []
    per_slice: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    errors: list[dict[str, Any]] = []

    for i, s in enumerate(samples):
        if ctx.cancelled:
            break
        try:
            path = Path(s.audio.uri.removeprefix("file://"))
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            hyp = asr(audio, sr)
        except Exception as e:
            ctx.log(f"  sample {s.id}: {e}")
            continue
        refs.append(s.transcript or "")
        hyps.append(hyp)
        processed_samples.append(s)
        for key in slice_by:
            v = getattr(s, key, None) or (s.metadata or {}).get(key)
            if v:
                per_slice[key][str(v)].append((s.transcript or "", hyp))
        if i % 25 == 0:
            ctx.log(f"  step processed={i}")
            ctx.heartbeat()

    overall = float(jiwer.wer(refs, hyps)) if refs else 1.0
    cer = float(jiwer.cer(refs, hyps)) if refs else 1.0
    ctx.log(f"step overall_wer={overall:.4f} cer={cer:.4f}")

    slices: dict[str, dict[str, dict[str, float | int]]] = {}
    for key, buckets in per_slice.items():
        slices[key] = {
            bucket: {
                "wer": float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])),
                "n": len(pairs),
            }
            for bucket, pairs in buckets.items()
        }

    # Top-10 worst examples by edit distance.
    for smpl, r, h in zip(processed_samples, refs, hyps, strict=True):
        if r and h:
            measures = jiwer.process_words(r, h)
            errors.append(
                {
                    "id": smpl.id,
                    "audio_uri": smpl.audio.uri,
                    "ref": r,
                    "hyp": h,
                    "wer": float(measures.wer),
                    "subs": int(measures.substitutions),
                    "ins": int(measures.insertions),
                    "del": int(measures.deletions),
                }
            )
    errors.sort(key=lambda x: x["wer"], reverse=True)

    return {
        "wer": overall,
        "cer": cer,
        "n_samples": len(refs),
        "slices": slices,
        "worst": errors[:10],
    }


def _load_asr(artifact_path: Path, language: str) -> Any:
    target = str(artifact_path)
    try:
        from faster_whisper import WhisperModel

        size = target.split("/")[-1].replace("whisper-", "")
        model = WhisperModel(size, compute_type="int8")

        def fn(audio: Any, sr: int) -> str:
            segments, _ = model.transcribe(audio, language=language, beam_size=1)
            return " ".join(seg.text.strip() for seg in segments).strip()

        return fn
    except Exception:
        from transformers import pipeline

        pipe = pipeline("automatic-speech-recognition", model=target)

        def fn(audio: Any, sr: int) -> str:
            return str(pipe({"array": audio, "sampling_rate": sr}).get("text", "")).strip()

        return fn


with contextlib.suppress(ValueError):
    register_handler("asr_eval", asr_eval_handler)
