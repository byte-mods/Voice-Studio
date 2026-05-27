"""Job handler: evaluate a TTS ModelVersion.

For v1 we report a **round-trip WER**: synthesize the reference text, transcribe
the result with Whisper, compare. This is a reliable proxy for intelligibility
and surfaces pronunciation issues. MOS estimators (UTMOS / NISQA) and speaker-
similarity will be added when those packages are pinned into `[tts_eval]`.

Config:

    {
      "model_version_id": "...",
      "dataset_version_id": "...",
      "split": "test",
      "max_samples": 50,
      "asr_model": "openai/whisper-tiny",
      "language": "en"
    }
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any

from oas_core.db import DatasetVersion, ModelVersion, session_scope
from oas_core.manifest import ManifestReader, Modality, TTSSample
from oas_core.queue.backend import JobContext, register_handler

log = logging.getLogger(__name__)


def _load_tts_split(manifest_root: Path, split: str) -> list[TTSSample]:
    out: list[TTSSample] = []
    for s in ManifestReader(manifest_root):
        if s.modality != Modality.TTS:
            continue
        if s.split.value != split:
            continue
        out.append(s)
    return out


def tts_eval_handler(ctx: JobContext) -> dict[str, Any]:
    import jiwer
    import numpy as np

    cfg = ctx.config
    mv_id = cfg["model_version_id"]
    dv_id = cfg["dataset_version_id"]
    split = cfg.get("split", "test")
    max_samples = int(cfg.get("max_samples", 50))
    asr_model_id = cfg.get("asr_model", "openai/whisper-tiny")
    language = cfg.get("language", "en")

    with session_scope() as db:
        mv = db.get(ModelVersion, mv_id)
        dv = db.get(DatasetVersion, dv_id)
        if not mv or not dv:
            raise ValueError("model_version_id or dataset_version_id not found")
        artifact_path = Path(mv.artifact_uri.removeprefix("file://"))
        manifest_root = Path(dv.manifest_uri.removeprefix("file://"))
        fmt = mv.format

    samples = _load_tts_split(manifest_root, split)[:max_samples]
    if not samples:
        raise ValueError(f"no TTS samples in split {split!r}")
    ctx.log(f"evaluating on {len(samples)} samples")

    synth = _load_tts(artifact_path, fmt)
    asr = _load_asr(asr_model_id, language)

    refs: list[str] = []
    hyps: list[str] = []
    durations: list[float] = []

    for i, s in enumerate(samples):
        if ctx.cancelled:
            break
        try:
            audio, sr = synth(s.text)
            hyp = asr(audio, sr)
        except Exception as e:
            ctx.log(f"  sample {s.id}: {e}")
            continue
        refs.append(s.text)
        hyps.append(hyp)
        durations.append(float(len(audio)) / float(sr))
        if i % 5 == 0:
            ctx.log(f"  step processed={i}")
            ctx.heartbeat()

    wer = float(jiwer.wer(refs, hyps)) if refs else 1.0
    total_audio = float(np.sum(durations)) if durations else 0.0
    ctx.log(f"step round_trip_wer={wer:.4f} mean_duration_s={total_audio / max(len(durations), 1):.3f}")
    return {
        "round_trip_wer": wer,
        "n_samples": len(refs),
        "mean_duration_s": total_audio / max(len(durations), 1),
        "total_audio_s": total_audio,
    }


def _load_tts(artifact_path: Path, fmt: str) -> Any:
    import numpy as np

    if fmt == "piper-onnx":
        from piper import PiperVoice

        onnx = next(artifact_path.glob("*.onnx"))
        voice = PiperVoice.load(str(onnx))

        def fn(text: str) -> tuple[Any, int]:
            buf = io.BytesIO()
            voice.synthesize(text, buf)
            buf.seek(0)
            import soundfile as sf

            data, sr = sf.read(buf, dtype="float32", always_2d=False)
            return data.astype(np.float32), int(sr)

        return fn

    from transformers import pipeline

    pipe = pipeline("text-to-speech", model=str(artifact_path))

    def fn_hf(text: str) -> tuple[Any, int]:
        out = pipe(text)
        audio = np.asarray(out["audio"], dtype=np.float32).squeeze()
        return audio, int(out["sampling_rate"])

    return fn_hf


def _load_asr(model_id: str, language: str) -> Any:
    try:
        from faster_whisper import WhisperModel

        size = model_id.split("/")[-1].replace("whisper-", "")
        model = WhisperModel(size, compute_type="int8")

        def fn(audio: Any, sr: int) -> str:
            segments, _ = model.transcribe(audio, language=language, beam_size=1)
            return " ".join(seg.text.strip() for seg in segments).strip()

        return fn
    except Exception:
        from transformers import pipeline

        pipe = pipeline("automatic-speech-recognition", model=model_id)

        def fn(audio: Any, sr: int) -> str:
            return str(pipe({"array": audio, "sampling_rate": sr}).get("text", "")).strip()

        return fn


with contextlib.suppress(ValueError):
    register_handler("tts_eval", tts_eval_handler)
