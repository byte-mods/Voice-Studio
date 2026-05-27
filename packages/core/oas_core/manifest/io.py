"""Streaming readers and writers for manifests.

Disk layout for a dataset version:

    <root>/
      manifest.json        # ManifestHeader (JSON)
      samples.jsonl        # one Sample per line (canonical)
      samples.parquet      # optional columnar mirror

Why JSONL is canonical: it is append-friendly, diff-friendly, and survives
partial writes. Parquet is generated on demand for analytics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

import orjson
from pydantic import TypeAdapter

from oas_core.manifest.schema import (
    Manifest,
    ManifestHeader,
    ManifestStats,
    Sample,
)

_SAMPLE_ADAPTER: TypeAdapter[Sample] = TypeAdapter(Sample)


def _dumps(obj: object) -> bytes:
    return orjson.dumps(obj, option=orjson.OPT_UTC_Z | orjson.OPT_NON_STR_KEYS)


class ManifestWriter:
    """Append samples to a manifest directory without holding them in memory.

    Use as a context manager::

        with ManifestWriter(path, header) as w:
            for sample in iter_samples():
                w.add(sample)
    """

    def __init__(self, root: str | Path, header: ManifestHeader) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.header = header
        self._jsonl_path = self.root / "samples.jsonl"
        self._header_path = self.root / "manifest.json"
        self._fp: IO[bytes] | None = None
        self._stats = ManifestStats()
        self._speakers: set[str] = set()

    def __enter__(self) -> ManifestWriter:
        self._fp = self._jsonl_path.open("ab")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, sample: Sample) -> None:
        if self._fp is None:
            raise RuntimeError("ManifestWriter not opened; use as a context manager")
        payload = _SAMPLE_ADAPTER.dump_python(sample, mode="json")
        self._fp.write(_dumps(payload))
        self._fp.write(b"\n")
        self._update_stats(sample)

    def extend(self, samples: Iterable[Sample]) -> None:
        for s in samples:
            self.add(s)

    def _update_stats(self, sample: Sample) -> None:
        s = self._stats
        s.num_samples += 1
        s.by_split[sample.split.value] = s.by_split.get(sample.split.value, 0) + 1
        if sample.language:
            s.by_language[sample.language] = s.by_language.get(sample.language, 0) + 1
        # Audio + speaker stats per modality
        audio = getattr(sample, "audio", None)
        if audio is not None:
            s.total_audio_s += float(audio.duration_s)
        speaker_id = getattr(sample, "speaker_id", None)
        if speaker_id:
            self._speakers.add(speaker_id)
        for turn in getattr(sample, "turns", []) or []:
            if turn.audio is not None:
                s.total_audio_s += float(turn.audio.duration_s)
            if turn.speaker_id:
                self._speakers.add(turn.speaker_id)

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        self._stats.num_speakers = len(self._speakers)
        self.header.stats = self._stats
        self._header_path.write_bytes(
            _dumps(self.header.model_dump(mode="json"))
        )


class ManifestReader:
    """Stream a manifest from disk without loading every sample into RAM."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._header_path = self.root / "manifest.json"
        self._jsonl_path = self.root / "samples.jsonl"
        if not self._header_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {self.root}")
        self.header = ManifestHeader.model_validate_json(self._header_path.read_text())

    def __iter__(self) -> Iterator[Sample]:
        if not self._jsonl_path.exists():
            return
        with self._jsonl_path.open("rb") as fp:
            for line in fp:
                if not line.strip():
                    continue
                yield _SAMPLE_ADAPTER.validate_python(orjson.loads(line))

    def load(self) -> Manifest:
        """Load the whole manifest into memory. Use for small datasets only."""
        return Manifest(header=self.header, samples=list(self))


def write_manifest(root: str | Path, manifest: Manifest) -> None:
    with ManifestWriter(root, manifest.header) as w:
        w.extend(manifest.samples)


def read_manifest(root: str | Path) -> Manifest:
    return ManifestReader(root).load()


def update_sample(root: str | Path, sample_id: str, patch: dict[str, object]) -> Sample | None:
    """Apply a shallow patch to one sample identified by id, rewriting the
    JSONL file atomically. Returns the updated sample or None if not found.

    This is intentionally simple (full-file rewrite) — good for the curation
    workflow on datasets up to ~1M samples. For larger manifests, a sharded
    layout will replace this.
    """
    root = Path(root)
    jsonl = root / "samples.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(jsonl)

    tmp = jsonl.with_suffix(".jsonl.tmp")
    updated: Sample | None = None
    new_stats = ManifestStats()
    speakers: set[str] = set()

    with jsonl.open("rb") as fin, tmp.open("wb") as fout:
        for line in fin:
            if not line.strip():
                continue
            payload = orjson.loads(line)
            if payload.get("id") == sample_id:
                payload.update(patch)
                sample = _SAMPLE_ADAPTER.validate_python(payload)
                updated = sample
                fout.write(_dumps(_SAMPLE_ADAPTER.dump_python(sample, mode="json")))
                fout.write(b"\n")
            else:
                sample = _SAMPLE_ADAPTER.validate_python(payload)
                fout.write(line if line.endswith(b"\n") else line + b"\n")
            new_stats.num_samples += 1
            new_stats.by_split[sample.split.value] = new_stats.by_split.get(sample.split.value, 0) + 1
            if sample.language:
                new_stats.by_language[sample.language] = new_stats.by_language.get(sample.language, 0) + 1
            audio = getattr(sample, "audio", None)
            if audio is not None:
                new_stats.total_audio_s += float(audio.duration_s)
            speaker_id = getattr(sample, "speaker_id", None)
            if speaker_id:
                speakers.add(speaker_id)

    tmp.replace(jsonl)

    # Refresh header stats so downstream readers stay consistent.
    header_path = root / "manifest.json"
    if header_path.exists():
        header = ManifestHeader.model_validate_json(header_path.read_text())
        new_stats.num_speakers = len(speakers)
        header.stats = new_stats
        header_path.write_bytes(_dumps(header.model_dump(mode="json")))

    return updated


def init_manifest(root: str | Path, header: ManifestHeader) -> None:
    """Create an empty manifest directory with the given header.

    Writes `manifest.json` (overwriting any existing header) and ensures
    `samples.jsonl` exists (created empty if missing, **never truncated** —
    callers that need a fresh dataset version are expected to pass an empty
    `root` per the dataset router's pre-check).

    Why not just use `ManifestWriter`: the interactive dataset builders need a
    persisted header *before* any sample arrives, so the version row can be
    inserted into the DB and the user can start appending one sample at a time
    via the API. `ManifestWriter` only flushes the header on `close()`.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_bytes(_dumps(header.model_dump(mode="json")))
    jsonl = root / "samples.jsonl"
    if not jsonl.exists():
        jsonl.touch()


def append_sample(root: str | Path, sample: Sample) -> None:
    """Append one validated sample to an existing manifest and refresh header stats.

    The header is read, stats are incremented for the new sample, and the
    header is written back. `num_speakers` is recomputed by scanning the JSONL
    — accurate but O(n); acceptable for the interactive-builder use case
    (manifests up to ~1M samples, matching the same constraint documented on
    `update_sample`). For larger workloads, a sharded layout will replace this.

    Raises `FileNotFoundError` if the manifest directory is uninitialized
    (no `manifest.json`).
    """
    root = Path(root)
    header_path = root / "manifest.json"
    if not header_path.exists():
        raise FileNotFoundError(header_path)
    jsonl_path = root / "samples.jsonl"

    payload = _SAMPLE_ADAPTER.dump_python(sample, mode="json")
    with jsonl_path.open("ab") as fp:
        fp.write(_dumps(payload))
        fp.write(b"\n")

    header = ManifestHeader.model_validate_json(header_path.read_text())
    stats = header.stats
    stats.num_samples += 1
    stats.by_split[sample.split.value] = stats.by_split.get(sample.split.value, 0) + 1
    if sample.language:
        stats.by_language[sample.language] = stats.by_language.get(sample.language, 0) + 1
    audio = getattr(sample, "audio", None)
    if audio is not None:
        stats.total_audio_s += float(audio.duration_s)
    for turn in getattr(sample, "turns", []) or []:
        if turn.audio is not None:
            stats.total_audio_s += float(turn.audio.duration_s)

    # num_speakers needs the full set across the manifest; recompute by scan.
    speakers: set[str] = set()
    with jsonl_path.open("rb") as fp:
        for line in fp:
            if not line.strip():
                continue
            sd = orjson.loads(line)
            sid = sd.get("speaker_id")
            if sid:
                speakers.add(sid)
            for t in sd.get("turns") or []:
                tsid = t.get("speaker_id")
                if tsid:
                    speakers.add(tsid)
    stats.num_speakers = len(speakers)

    header.stats = stats
    header_path.write_bytes(_dumps(header.model_dump(mode="json")))


def export_json_schema() -> dict[str, object]:
    """Return the JSON Schema for a single Sample, for use by the web UI / SDK."""
    return _SAMPLE_ADAPTER.json_schema()


if __name__ == "__main__":
    import sys

    print(json.dumps(export_json_schema(), indent=2))
    sys.exit(0)
