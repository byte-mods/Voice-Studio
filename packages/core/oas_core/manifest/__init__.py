"""Dataset manifest v1.

A manifest is a typed, append-friendly list of samples. Each sample belongs to
one of four modalities — ASR, TTS, LLM, S2S — and carries enough metadata for
training, evaluation, licensing, and reproducibility.

Manifests are stored on disk as:
- `manifest.json`  — header with schema_version, modality, splits, stats.
- `samples.jsonl`  — one sample per line (streaming-friendly).
- `samples.parquet` — columnar mirror for fast analytics (optional).
"""

from oas_core.manifest.io import (
    ManifestReader,
    ManifestWriter,
    append_sample,
    init_manifest,
    read_manifest,
    update_sample,
    write_manifest,
)
from oas_core.manifest.schema import (
    SCHEMA_VERSION,
    ASRSample,
    AudioRef,
    ConsentRecord,
    DialogTurn,
    LicenseInfo,
    LLMSample,
    Manifest,
    ManifestHeader,
    ManifestStats,
    Modality,
    S2SSample,
    SampleBase,
    Split,
    TTSSample,
)

__all__ = [
    "SCHEMA_VERSION",
    "ASRSample",
    "AudioRef",
    "ConsentRecord",
    "DialogTurn",
    "LLMSample",
    "LicenseInfo",
    "Manifest",
    "ManifestHeader",
    "ManifestReader",
    "ManifestStats",
    "ManifestWriter",
    "Modality",
    "S2SSample",
    "SampleBase",
    "Split",
    "TTSSample",
    "append_sample",
    "init_manifest",
    "read_manifest",
    "update_sample",
    "write_manifest",
]
