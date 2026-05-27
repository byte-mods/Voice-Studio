from pathlib import Path

from oas_core.manifest import (
    ASRSample,
    AudioRef,
    DialogTurn,
    LicenseInfo,
    LLMSample,
    Manifest,
    ManifestHeader,
    ManifestReader,
    ManifestWriter,
    Modality,
    S2SSample,
    Split,
    TTSSample,
)
from oas_core.manifest.schema import Role


def _license() -> LicenseInfo:
    return LicenseInfo(spdx="CC-BY-4.0", holder="Test")


def _audio(duration: float = 1.5) -> AudioRef:
    return AudioRef(
        uri="file:///tmp/x.wav", sample_rate=16000, channels=1, duration_s=duration
    )


def test_asr_sample_roundtrip(tmp_path: Path) -> None:
    sample = ASRSample(
        language="en", license=_license(), audio=_audio(2.0), transcript="hello world"
    )
    header = ManifestHeader(dataset_id="ds1", name="t", modality=Modality.ASR)
    with ManifestWriter(tmp_path, header) as w:
        w.add(sample)
    reader = ManifestReader(tmp_path)
    out = list(reader)
    assert len(out) == 1
    assert out[0].id == sample.id
    assert reader.header.stats.num_samples == 1
    assert reader.header.stats.total_audio_s == 2.0


def test_all_modalities_write(tmp_path: Path) -> None:
    samples = [
        ASRSample(language="en", license=_license(), audio=_audio(), transcript="a"),
        TTSSample(
            language="en", license=_license(), audio=_audio(), text="a", speaker_id="spk1"
        ),
        LLMSample(
            language="en",
            license=_license(),
            turns=[DialogTurn(role=Role.USER, text="hi")],
        ),
        S2SSample(
            language="en",
            license=_license(),
            turns=[
                DialogTurn(role=Role.USER, text="hi", audio=_audio()),
                DialogTurn(role=Role.ASSISTANT, text="hello", audio=_audio()),
            ],
        ),
    ]
    header = ManifestHeader(dataset_id="mix", name="mix", modality=Modality.S2S)
    with ManifestWriter(tmp_path, header) as w:
        for s in samples:
            w.add(s)
    out = list(ManifestReader(tmp_path))
    assert {s.modality for s in out} == {Modality.ASR, Modality.TTS, Modality.LLM, Modality.S2S}


def test_splits_counted(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d", name="d", modality=Modality.ASR)
    with ManifestWriter(tmp_path, header) as w:
        w.add(ASRSample(license=_license(), audio=_audio(), transcript="a", split=Split.TRAIN))
        w.add(ASRSample(license=_license(), audio=_audio(), transcript="b", split=Split.VAL))
        w.add(ASRSample(license=_license(), audio=_audio(), transcript="c", split=Split.VAL))
    reader = ManifestReader(tmp_path)
    assert reader.header.stats.by_split == {"train": 1, "val": 2}


def test_in_memory_manifest_roundtrip(tmp_path: Path) -> None:
    m = Manifest(
        header=ManifestHeader(dataset_id="d", name="d", modality=Modality.ASR),
        samples=[ASRSample(license=_license(), audio=_audio(), transcript="x")],
    )
    from oas_core.manifest import read_manifest, write_manifest

    write_manifest(tmp_path, m)
    loaded = read_manifest(tmp_path)
    assert len(loaded.samples) == 1
    assert loaded.samples[0].transcript == "x"
