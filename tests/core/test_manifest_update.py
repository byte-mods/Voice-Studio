from pathlib import Path

from oas_core.manifest import (
    ASRSample,
    AudioRef,
    LicenseInfo,
    ManifestHeader,
    ManifestReader,
    ManifestWriter,
    Modality,
    update_sample,
)


def _audio() -> AudioRef:
    return AudioRef(uri="file:///fake.wav", sample_rate=16000, duration_s=1.0)


def test_update_sample_rewrites_one_line_only(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d", name="d", modality=Modality.ASR)
    with ManifestWriter(tmp_path, header) as w:
        s1 = ASRSample(license=LicenseInfo(spdx="CC0-1.0"), audio=_audio(), transcript="a")
        s2 = ASRSample(license=LicenseInfo(spdx="CC0-1.0"), audio=_audio(), transcript="b")
        s3 = ASRSample(license=LicenseInfo(spdx="CC0-1.0"), audio=_audio(), transcript="c")
        for s in (s1, s2, s3):
            w.add(s)
        ids = [s.id for s in (s1, s2, s3)]

    updated = update_sample(tmp_path, ids[1], {"transcript": "B_FIXED"})
    assert updated is not None
    assert updated.transcript == "B_FIXED"

    out = list(ManifestReader(tmp_path))
    assert [s.transcript for s in out] == ["a", "B_FIXED", "c"]


def test_update_sample_unknown_id(tmp_path: Path) -> None:
    header = ManifestHeader(dataset_id="d", name="d", modality=Modality.ASR)
    with ManifestWriter(tmp_path, header) as w:
        w.add(ASRSample(license=LicenseInfo(spdx="CC0-1.0"), audio=_audio(), transcript="a"))
    assert update_sample(tmp_path, "no-such-id", {"transcript": "x"}) is None
