"""Job handler: import a Hugging Face dataset into a manifest v1 dataset version.

Config schema (`Job.config`):

    {
      "hf_id": "mozilla-foundation/common_voice_16_1",   # required
      "hf_config": "en",                                  # optional
      "hf_split": "train",                                # default 'train'
      "modality": "asr",                                  # required
      "dataset_id": "<oas Dataset.id>",                   # required
      "version": "0.1.0",                                 # required (semver)
      "language": "en",                                   # optional default
      "license": {"spdx": "CC0-1.0"},                     # required for OAS
      "field_map": {                                      # maps HF cols -> sample fields
          "audio": "audio",
          "transcript": "sentence",
          "speaker_id": "client_id"
      },
      "max_samples": 1000,                                # optional cap
      "streaming": true                                   # default true
    }

The handler:
  1. Streams the HF dataset.
  2. Materializes audio bytes to LocalStorage (file:// URIs).
  3. Builds typed Sample objects via field mapping.
  4. Writes them through ManifestWriter into <data_dir>/datasets/<dv_id>/.
  5. Registers a DatasetVersion row pointing at the manifest dir.

This handler imports `datasets` lazily so the core server keeps a small footprint
when no HF imports are running.
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any

from oas_core.db import Dataset, DatasetVersion, session_scope
from oas_core.manifest import (
    ASRSample,
    AudioRef,
    DialogTurn,
    LicenseInfo,
    LLMSample,
    ManifestHeader,
    ManifestWriter,
    Modality,
    S2SSample,
    Split,
    TTSSample,
)
from oas_core.manifest.schema import Role
from oas_core.queue.backend import JobContext, register_handler
from oas_core.settings import get_settings
from oas_core.storage import LocalStorage

log = logging.getLogger(__name__)


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"hf_import config missing required key: {key!r}")
    return d[key]


def _materialize_audio(
    storage: LocalStorage, dv_id: str, idx: int, hf_audio: dict[str, Any]
) -> AudioRef:
    """Persist a HF audio dict ({'array': np.ndarray, 'sampling_rate': int} or
    {'bytes': b'...'} or {'path': '...'}) to local storage and return AudioRef."""
    import soundfile as sf  # lazy

    rel = f"hf/{dv_id}/{idx:08d}.wav"
    uri = storage.uri_for(rel)

    if "array" in hf_audio and "sampling_rate" in hf_audio:
        buf = io.BytesIO()
        sf.write(buf, hf_audio["array"], hf_audio["sampling_rate"], format="WAV", subtype="PCM_16")
        data = buf.getvalue()
        sample_rate = int(hf_audio["sampling_rate"])
    elif "bytes" in hf_audio:
        data = hf_audio["bytes"]
        # Decode to probe sample rate
        info = sf.info(io.BytesIO(data))
        sample_rate = int(info.samplerate)
    elif hf_audio.get("path"):
        path = Path(hf_audio["path"])
        data = path.read_bytes()
        info = sf.info(io.BytesIO(data))
        sample_rate = int(info.samplerate)
    else:
        raise ValueError(f"Unsupported HF audio shape: keys={list(hf_audio)}")

    obj = storage.put_bytes(uri, data, content_type="audio/wav")
    duration = _probe_duration(data)
    return AudioRef(
        uri=uri,
        sample_rate=sample_rate,
        channels=1,
        duration_s=duration,
        sha256=obj.sha256,
    )


def _probe_duration(data: bytes) -> float:
    import soundfile as sf

    info = sf.info(io.BytesIO(data))
    return float(info.frames) / float(info.samplerate or 1)


def _build_sample(
    modality: Modality,
    record: dict[str, Any],
    field_map: dict[str, str],
    audio: AudioRef | None,
    license_info: LicenseInfo,
    language: str | None,
    source: str,
) -> Any:
    def pick(name: str, default: Any = None) -> Any:
        col = field_map.get(name)
        return record.get(col, default) if col else default

    # Typed Any-valued so `**common` unpack into each typed Sample constructor
    # passes mypy; Pydantic does the runtime field-shape validation.
    common: dict[str, Any] = dict(
        language=language,
        license=license_info,
        source=source,
        split=Split.TRAIN,
    )

    if modality == Modality.ASR:
        if audio is None:
            raise ValueError("ASR samples require non-None audio")
        return ASRSample(
            **common,
            audio=audio,
            transcript=str(pick("transcript", "")),
            speaker_id=pick("speaker_id"),
            accent=pick("accent"),
        )
    if modality == Modality.TTS:
        if audio is None:
            raise ValueError("TTS samples require non-None audio")
        return TTSSample(
            **common,
            audio=audio,
            text=str(pick("text", "")),
            speaker_id=str(pick("speaker_id", "unknown")),
            style=pick("style"),
        )
    if modality == Modality.LLM:
        # Expect a 'messages' field with [{role, content}, ...] or fall back to text.
        turns_raw = pick("messages")
        turns: list[DialogTurn]
        if isinstance(turns_raw, list):
            turns = [
                DialogTurn(role=Role(t.get("role", "user")), text=str(t.get("content", "")))
                for t in turns_raw
            ]
        else:
            turns = [DialogTurn(role=Role.USER, text=str(pick("transcript", "")))]
        return LLMSample(**common, turns=turns)
    if modality == Modality.S2S:
        # Minimal: a single user+assistant turn pair using mapped fields.
        turns = [
            DialogTurn(role=Role.USER, text=str(pick("transcript", "")), audio=audio),
            DialogTurn(role=Role.ASSISTANT, text=str(pick("response", ""))),
        ]
        return S2SSample(**common, turns=turns)
    raise ValueError(f"Unsupported modality {modality}")


def hf_import_handler(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    hf_id = _require(cfg, "hf_id")
    modality = Modality(_require(cfg, "modality"))
    dataset_id = _require(cfg, "dataset_id")
    version = _require(cfg, "version")
    license_info = LicenseInfo.model_validate(_require(cfg, "license"))
    field_map = dict(_require(cfg, "field_map"))

    hf_config = cfg.get("hf_config")
    hf_split = cfg.get("hf_split", "train")
    language = cfg.get("language")
    max_samples = cfg.get("max_samples")
    streaming = bool(cfg.get("streaming", True))

    settings = get_settings()
    storage = LocalStorage(settings.data_dir)

    # Use a stable dataset-version directory keyed by the run so concurrent
    # imports don't collide.
    dv_root = settings.datasets_dir / dataset_id / version
    dv_root.mkdir(parents=True, exist_ok=True)

    ctx.log(f"Importing HF dataset {hf_id!r} config={hf_config!r} split={hf_split!r} → {dv_root}")

    from datasets import load_dataset  # lazy

    ds = load_dataset(hf_id, hf_config, split=hf_split, streaming=streaming)

    header = ManifestHeader(
        dataset_id=dataset_id,
        dataset_version=version,
        name=f"{hf_id}:{hf_config or 'default'}",
        modality=modality,
        source=f"hf://{hf_id}",
        license_default=license_info,
    )

    count = 0
    with ManifestWriter(dv_root, header) as writer:
        for idx, record in enumerate(ds):
            if ctx.cancelled:
                ctx.log("cancelled mid-import")
                break
            if max_samples and count >= int(max_samples):
                break

            audio_ref: AudioRef | None = None
            audio_field = field_map.get("audio")
            if audio_field and audio_field in record and modality in (Modality.ASR, Modality.TTS, Modality.S2S):
                try:
                    audio_ref = _materialize_audio(storage, dataset_id + "_" + version, idx, record[audio_field])
                except Exception as e:
                    ctx.log(f"skipping idx={idx}: audio decode failed: {e}")
                    continue

            try:
                sample = _build_sample(
                    modality, record, field_map, audio_ref, license_info, language, f"hf://{hf_id}"
                )
            except Exception as e:
                ctx.log(f"skipping idx={idx}: sample build failed: {e}")
                continue

            writer.add(sample)
            count += 1
            if count % 100 == 0:
                ctx.log(f"imported {count} samples")
                ctx.heartbeat()

    manifest_uri = f"file://{dv_root}"
    ctx.log(f"wrote {count} samples to {manifest_uri}")

    # Register DatasetVersion in the DB.
    with session_scope() as s:
        if not s.get(Dataset, dataset_id):
            raise ValueError(f"Dataset {dataset_id!r} not found — was it deleted?")
        existing = (
            s.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.version == version)
            .first()
        )
        if existing:
            existing.manifest_uri = manifest_uri
            existing.num_samples = count
        else:
            s.add(
                DatasetVersion(
                    dataset_id=dataset_id,
                    version=version,
                    manifest_uri=manifest_uri,
                    num_samples=count,
                    notes=f"Imported from {hf_id}",
                )
            )

    return {"hf_id": hf_id, "samples_imported": count, "manifest_uri": manifest_uri}


with contextlib.suppress(ValueError):
    register_handler("hf_import", hf_import_handler)
