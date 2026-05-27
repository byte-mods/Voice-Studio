"""Job handler: bootstrap an ASR dataset from raw audio.

Config:

    {
      "audio_uris": ["file://...wav", ...],   # required
      "dataset_id": "...",                     # required
      "version": "0.1.0",                      # required
      "language": "en",                        # default 'en'
      "license": {"spdx": "..."},              # required
      "whisper_model": "openai/whisper-tiny",  # default
      "vad": {                                 # silero VAD options
        "min_silence_ms": 400,
        "min_speech_ms": 250,
        "threshold": 0.5
      }
    }

Pipeline:
  1. Load each source audio with soundfile, resample to 16 kHz mono.
  2. Run Silero VAD to detect speech segments.
  3. Slice each segment to its own file in storage.
  4. Transcribe each segment with faster-whisper (or transformers Whisper as fallback).
  5. Write ASRSample entries through ManifestWriter.
  6. Register a DatasetVersion.

All heavy deps are imported lazily so the studio runs without them installed.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oas_core.db import Dataset, DatasetVersion, session_scope

if TYPE_CHECKING:
    import numpy as np
import contextlib

from oas_core.manifest import (
    ASRSample,
    AudioRef,
    LicenseInfo,
    ManifestHeader,
    ManifestWriter,
    Modality,
)
from oas_core.queue.backend import JobContext, register_handler
from oas_core.settings import get_settings
from oas_core.storage import LocalStorage

log = logging.getLogger(__name__)

TARGET_SR = 16000


def _load_audio_mono16k(path: Path) -> np.ndarray:
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        except ImportError as e:
            raise RuntimeError(
                f"audio is {sr} Hz; install 'librosa' to resample to {TARGET_SR} Hz"
            ) from e
    return data.astype(np.float32)


def _vad_segments(audio: np.ndarray, opts: dict[str, Any]) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) speech segments via Silero VAD."""
    import torch

    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
    )
    (get_speech_timestamps, _, _, _, _) = utils
    ts = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=TARGET_SR,
        min_silence_duration_ms=int(opts.get("min_silence_ms", 400)),
        min_speech_duration_ms=int(opts.get("min_speech_ms", 250)),
        threshold=float(opts.get("threshold", 0.5)),
    )
    return [(t["start"] / TARGET_SR, t["end"] / TARGET_SR) for t in ts]


class _Transcriber:
    """Wraps faster-whisper with a transformers fallback."""

    def __init__(self, model_id: str, language: str | None) -> None:
        self.language = language
        self._fw = None
        self._hf = None
        try:
            from faster_whisper import WhisperModel

            size = model_id.split("/")[-1].replace("whisper-", "")
            self._fw = WhisperModel(size, compute_type="int8")
        except Exception:
            from transformers import pipeline

            self._hf = pipeline("automatic-speech-recognition", model=model_id)

    def __call__(self, audio: np.ndarray) -> str:
        if self._fw is not None:
            segments, _ = self._fw.transcribe(audio, language=self.language)
            return " ".join(seg.text.strip() for seg in segments).strip()
        assert self._hf is not None
        out = self._hf({"array": audio, "sampling_rate": TARGET_SR})
        return str(out.get("text", "")).strip()


def asr_bootstrap_handler(ctx: JobContext) -> dict[str, Any]:
    import soundfile as sf

    cfg = ctx.config
    audio_uris: list[str] = cfg["audio_uris"]
    dataset_id = cfg["dataset_id"]
    version = cfg["version"]
    license_info = LicenseInfo.model_validate(cfg["license"])
    language = cfg.get("language", "en")
    whisper_model = cfg.get("whisper_model", "openai/whisper-tiny")
    vad_opts = cfg.get("vad", {})

    settings = get_settings()
    storage = LocalStorage(settings.data_dir)

    dv_root = settings.datasets_dir / dataset_id / version
    dv_root.mkdir(parents=True, exist_ok=True)
    segments_root = settings.data_dir / "segments" / dataset_id / version
    segments_root.mkdir(parents=True, exist_ok=True)

    ctx.log(f"bootstrapping ASR dataset {dataset_id} v{version} from {len(audio_uris)} sources")
    ctx.log(f"loading transcriber: {whisper_model}")
    transcriber = _Transcriber(whisper_model, language)

    header = ManifestHeader(
        dataset_id=dataset_id,
        dataset_version=version,
        name=f"bootstrap:{dataset_id}",
        modality=Modality.ASR,
        license_default=license_info,
        source="asr_bootstrap",
    )

    total_segments = 0
    with ManifestWriter(dv_root, header) as writer:
        for src_idx, uri in enumerate(audio_uris):
            if ctx.cancelled:
                ctx.log("cancelled")
                break
            path = Path(uri.removeprefix("file://"))
            ctx.log(f"[{src_idx + 1}/{len(audio_uris)}] {path.name}")

            try:
                audio = _load_audio_mono16k(path)
            except Exception as e:
                ctx.log(f"  load failed: {e}")
                continue

            try:
                segs = _vad_segments(audio, vad_opts)
            except Exception as e:
                ctx.log(f"  VAD failed: {e}; treating whole file as one segment")
                segs = [(0.0, len(audio) / TARGET_SR)]

            ctx.log(f"  found {len(segs)} speech segments")

            for seg_idx, (start_s, end_s) in enumerate(segs):
                if ctx.cancelled:
                    break
                start_frame = int(start_s * TARGET_SR)
                end_frame = int(end_s * TARGET_SR)
                clip = audio[start_frame:end_frame]
                if clip.size == 0:
                    continue

                buf = io.BytesIO()
                sf.write(buf, clip, TARGET_SR, format="WAV", subtype="PCM_16")
                seg_rel = f"segments/{dataset_id}/{version}/{src_idx:04d}_{seg_idx:06d}.wav"
                seg_uri = storage.uri_for(seg_rel)
                storage.put_bytes(seg_uri, buf.getvalue(), content_type="audio/wav")

                try:
                    transcript = transcriber(clip)
                except Exception as e:
                    ctx.log(f"  transcribe failed for seg {seg_idx}: {e}")
                    transcript = ""

                writer.add(
                    ASRSample(
                        license=license_info,
                        language=language,
                        audio=AudioRef(
                            uri=seg_uri,
                            sample_rate=TARGET_SR,
                            channels=1,
                            duration_s=float(end_s - start_s),
                        ),
                        transcript=transcript,
                        metadata={
                            "source_uri": uri,
                            "source_start_s": float(start_s),
                            "source_end_s": float(end_s),
                            "auto_transcribed_by": whisper_model,
                            "needs_review": True,
                        },
                    )
                )
                total_segments += 1
                if total_segments % 25 == 0:
                    ctx.log(f"  written {total_segments} segments")
                    ctx.heartbeat()

    manifest_uri = f"file://{dv_root}"
    ctx.log(f"wrote {total_segments} segments to {manifest_uri}")

    with session_scope() as s:
        if not s.get(Dataset, dataset_id):
            raise ValueError(f"Dataset {dataset_id!r} not found")
        existing = (
            s.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == version)
            .first()
        )
        if existing:
            existing.manifest_uri = manifest_uri
            existing.num_samples = total_segments
        else:
            s.add(
                DatasetVersion(
                    dataset_id=dataset_id,
                    version=version,
                    manifest_uri=manifest_uri,
                    num_samples=total_segments,
                    notes=f"VAD+Whisper bootstrap from {len(audio_uris)} source(s)",
                )
            )

    return {"segments": total_segments, "manifest_uri": manifest_uri}


with contextlib.suppress(ValueError):
    register_handler("asr_bootstrap", asr_bootstrap_handler)
