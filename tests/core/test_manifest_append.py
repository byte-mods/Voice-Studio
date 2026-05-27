"""Tests for `init_manifest` and `append_sample`.

These cover the interactive-builder path: create an empty manifest version
on disk, then append one sample at a time via the dataset router's
`/datasets/versions/{vid}/samples` endpoint.
"""

from pathlib import Path

import pytest
from oas_core.manifest import (
    ASRSample,
    AudioRef,
    DialogTurn,
    LicenseInfo,
    ManifestHeader,
    ManifestReader,
    Modality,
    S2SSample,
    Split,
    append_sample,
    init_manifest,
)


def _license() -> LicenseInfo:
    return LicenseInfo(spdx="CC0-1.0", holder="Test")


def _audio(duration: float = 1.5) -> AudioRef:
    return AudioRef(uri="file:///tmp/x.wav", sample_rate=16000, channels=1, duration_s=duration)


def test_init_manifest_creates_header_and_empty_jsonl(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d1", name="d1", modality=Modality.ASR)
    init_manifest(tmp_path, header)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "samples.jsonl").exists()
    assert (tmp_path / "samples.jsonl").read_bytes() == b""

    # Header roundtrips through the reader.
    reader = ManifestReader(tmp_path)
    assert reader.header.dataset_id == "d1"
    assert reader.header.stats.num_samples == 0
    assert list(reader) == []


def test_init_manifest_idempotent_does_not_truncate_jsonl(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d1", name="d1", modality=Modality.ASR)
    init_manifest(tmp_path, header)
    # Simulate a prior write.
    (tmp_path / "samples.jsonl").write_bytes(b"existing line\n")
    # Re-init with a new header must overwrite the header but leave the jsonl alone.
    header2 = ManifestHeader(dataset_id="d1", name="d1-renamed", modality=Modality.ASR)
    init_manifest(tmp_path, header2)
    assert (tmp_path / "samples.jsonl").read_bytes() == b"existing line\n"
    assert ManifestReader(tmp_path).header.name == "d1-renamed"


def test_append_sample_adds_one_and_refreshes_stats(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d1", name="d1", modality=Modality.ASR)
    init_manifest(tmp_path, header)

    s1 = ASRSample(
        language="en",
        license=_license(),
        audio=_audio(2.0),
        transcript="hello",
        speaker_id="spk1",
        split=Split.TRAIN,
    )
    append_sample(tmp_path, s1)

    reader = ManifestReader(tmp_path)
    assert reader.header.stats.num_samples == 1
    assert reader.header.stats.by_split == {"train": 1}
    assert reader.header.stats.by_language == {"en": 1}
    assert reader.header.stats.total_audio_s == pytest.approx(2.0)
    assert reader.header.stats.num_speakers == 1
    out = list(reader)
    assert len(out) == 1
    assert out[0].transcript == "hello"


def test_append_sample_accumulates(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d1", name="d1", modality=Modality.ASR)
    init_manifest(tmp_path, header)
    for i, lang, dur, spk, split in [
        (1, "en", 1.0, "spk1", Split.TRAIN),
        (2, "en", 2.5, "spk2", Split.TRAIN),
        (3, "es", 0.5, "spk1", Split.VAL),
    ]:
        append_sample(
            tmp_path,
            ASRSample(
                language=lang,
                license=_license(),
                audio=_audio(dur),
                transcript=f"line {i}",
                speaker_id=spk,
                split=split,
            ),
        )

    reader = ManifestReader(tmp_path)
    st = reader.header.stats
    assert st.num_samples == 3
    assert st.by_split == {"train": 2, "val": 1}
    assert st.by_language == {"en": 2, "es": 1}
    assert st.total_audio_s == pytest.approx(4.0)
    assert st.num_speakers == 2  # spk1, spk2


def test_append_sample_counts_s2s_turn_audio_and_speakers(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d1", name="d1", modality=Modality.S2S)
    init_manifest(tmp_path, header)

    s2s = S2SSample(
        license=_license(),
        turns=[
            DialogTurn(role="user", audio=_audio(1.0), speaker_id="user-1", text="hi"),
            DialogTurn(role="assistant", audio=_audio(2.0), speaker_id="asst-1", text="hello"),
        ],
    )
    append_sample(tmp_path, s2s)

    st = ManifestReader(tmp_path).header.stats
    assert st.num_samples == 1
    assert st.total_audio_s == pytest.approx(3.0)
    assert st.num_speakers == 2


def test_append_sample_raises_when_manifest_uninitialized(tmp_path: Path) -> None:
    s = ASRSample(license=_license(), audio=_audio(), transcript="x")
    with pytest.raises(FileNotFoundError):
        append_sample(tmp_path, s)
