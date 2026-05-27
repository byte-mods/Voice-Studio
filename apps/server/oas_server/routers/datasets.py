"""Dataset CRUD and version registration.

Dataset *content* (audio files, transcripts) lives in storage and is described
by manifest files. This router manages the *metadata* records that point to
those manifests, plus version stats.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from oas_core.db import Dataset, DatasetVersion, Project, Role, session_scope
from oas_core.db.models import Modality
from oas_core.manifest import (
    ManifestHeader,
    ManifestReader,
    ManifestWriter,
    Split,
    init_manifest,
    update_sample,
)
from oas_core.manifest import (
    append_sample as manifest_append_sample,
)
from oas_core.manifest.schema import (
    ASRSample,
    LicenseInfo,
    LLMSample,
    S2SSample,
    TTSSample,
)
from oas_core.settings import get_settings
from pydantic import BaseModel, Field, TypeAdapter
from sqlalchemy import select

from oas_server.auth import CurrentUser, assert_role, require_user

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetIn(BaseModel):
    project_id: str
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-_]*$")
    name: str
    modality: Modality
    description: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)


class DatasetOut(BaseModel):
    id: str
    project_id: str
    slug: str
    name: str
    modality: Modality
    description: str | None
    source: str | None
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class DatasetVersionIn(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    manifest_uri: str
    num_samples: int = 0
    total_audio_s: float = 0.0
    stats: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class DatasetVersionOut(BaseModel):
    id: str
    dataset_id: str
    version: str
    manifest_uri: str
    num_samples: int
    total_audio_s: float
    stats: dict[str, Any]
    notes: str | None
    created_at: datetime
    parent_version_id: str | None = None


def _to_dataset_out(d: Dataset) -> DatasetOut:
    return DatasetOut(
        id=d.id,
        project_id=d.project_id,
        slug=d.slug,
        name=d.name,
        modality=d.modality,
        description=d.description,
        source=d.source,
        tags=d.tags or [],
        created_at=d.created_at,
        updated_at=d.updated_at,
    )


def _to_version_out(v: DatasetVersion) -> DatasetVersionOut:
    return DatasetVersionOut(
        id=v.id,
        dataset_id=v.dataset_id,
        version=v.version,
        manifest_uri=v.manifest_uri,
        num_samples=v.num_samples,
        total_audio_s=v.total_audio_s,
        stats=v.stats or {},
        notes=v.notes,
        created_at=v.created_at,
        parent_version_id=v.parent_version_id,
    )


@router.get("", response_model=list[DatasetOut])
def list_datasets(project_id: str | None = None, modality: Modality | None = None) -> list[DatasetOut]:
    with session_scope() as s:
        stmt = select(Dataset)
        if project_id:
            stmt = stmt.where(Dataset.project_id == project_id)
        if modality:
            stmt = stmt.where(Dataset.modality == modality)
        return [_to_dataset_out(d) for d in s.scalars(stmt.order_by(Dataset.created_at.desc()))]


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
def create_dataset(body: DatasetIn, user: CurrentUser = Depends(require_user)) -> DatasetOut:
    assert_role(user, body.project_id, Role.EDITOR)
    with session_scope() as s:
        if not s.get(Project, body.project_id):
            raise HTTPException(404, "project not found")
        d = Dataset(
            project_id=body.project_id,
            slug=body.slug,
            name=body.name,
            modality=body.modality,
            description=body.description,
            source=body.source,
            tags=body.tags,
        )
        s.add(d)
        s.flush()
        return _to_dataset_out(d)


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str) -> DatasetOut:
    with session_scope() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404)
        return _to_dataset_out(d)


@router.get("/{dataset_id}/versions", response_model=list[DatasetVersionOut])
def list_versions(dataset_id: str) -> list[DatasetVersionOut]:
    with session_scope() as s:
        if not s.get(Dataset, dataset_id):
            raise HTTPException(404)
        stmt = (
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.created_at.desc())
        )
        return [_to_version_out(v) for v in s.scalars(stmt)]


class SamplePage(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[dict[str, Any]]


class SamplePatch(BaseModel):
    transcript: str | None = None
    text: str | None = None
    split: str | None = None
    language: str | None = None
    metadata: dict[str, Any] | None = None


@router.patch("/versions/{version_id}/samples/{sample_id}")
def patch_sample(
    version_id: str, sample_id: str, patch: SamplePatch, user: CurrentUser = Depends(require_user)
) -> dict[str, Any]:
    with session_scope() as s:
        v = s.get(DatasetVersion, version_id)
        if not v:
            raise HTTPException(404)
        d = s.get(Dataset, v.dataset_id)
        if d:
            assert_role(user, d.project_id, Role.EDITOR)
        root = Path(v.manifest_uri.removeprefix("file://"))

    payload = {k: v for k, v in patch.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(400, "empty patch")
    if "metadata" in payload:
        payload["metadata"] = {**payload["metadata"], "needs_review": False}

    updated = update_sample(root, sample_id, payload)
    if updated is None:
        raise HTTPException(404, "sample id not found in manifest")
    return updated.model_dump(mode="json")


@router.get("/versions/{version_id}/samples", response_model=SamplePage)
def list_samples(version_id: str, offset: int = 0, limit: int = 50) -> SamplePage:
    """Stream a page of samples from a dataset version's manifest.

    Reads the JSONL on disk so this stays O(offset + limit) instead of loading
    the full manifest into memory.
    """
    with session_scope() as s:
        v = s.get(DatasetVersion, version_id)
        if not v:
            raise HTTPException(404)
        manifest_root = Path(v.manifest_uri.removeprefix("file://"))

    if not (manifest_root / "manifest.json").exists():
        raise HTTPException(404, f"manifest not found at {manifest_root}")

    reader = ManifestReader(manifest_root)
    items: list[dict[str, Any]] = []
    for seen, sample in enumerate(reader):
        if seen >= offset + limit:
            break
        if seen >= offset:
            items.append(sample.model_dump(mode="json"))

    return SamplePage(
        total=reader.header.stats.num_samples,
        offset=offset,
        limit=limit,
        items=items,
    )


@router.post("/{dataset_id}/versions", response_model=DatasetVersionOut, status_code=201)
def add_version(
    dataset_id: str, body: DatasetVersionIn, user: CurrentUser = Depends(require_user)
) -> DatasetVersionOut:
    with session_scope() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404)
        assert_role(user, d.project_id, Role.EDITOR)
        v = DatasetVersion(
            dataset_id=dataset_id,
            version=body.version,
            manifest_uri=body.manifest_uri,
            num_samples=body.num_samples,
            total_audio_s=body.total_audio_s,
            stats=body.stats,
            notes=body.notes,
        )
        s.add(v)
        s.flush()
        return _to_version_out(v)


class InitVersionIn(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    license: dict[str, Any] = Field(
        default_factory=lambda: {"spdx": "CC0-1.0"},
        description="Default license for samples appended later.",
    )
    notes: str | None = None


@router.post("/{dataset_id}/versions/init", response_model=DatasetVersionOut, status_code=201)
def init_version(
    dataset_id: str, body: InitVersionIn, user: CurrentUser = Depends(require_user)
) -> DatasetVersionOut:
    """Create an empty manifest dir + DatasetVersion ready to be appended to.

    Use this when building a dataset interactively (chat / voice / conversation
    builders). Each `POST /datasets/versions/{vid}/samples` will append.
    """
    settings = get_settings()
    with session_scope() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404)
        assert_role(user, d.project_id, Role.EDITOR)

        root = settings.datasets_dir / dataset_id / body.version
        if root.exists() and any(root.iterdir()):
            raise HTTPException(409, f"version {body.version!r} already exists on disk")
        header = ManifestHeader(
            dataset_id=dataset_id,
            dataset_version=body.version,
            name=f"{d.slug}:{body.version}",
            modality=d.modality,
            license_default=LicenseInfo.model_validate(body.license),
        )
        init_manifest(root, header)

        v = DatasetVersion(
            dataset_id=dataset_id,
            version=body.version,
            manifest_uri=f"file://{root}",
            num_samples=0,
            notes=body.notes,
        )
        s.add(v)
        s.flush()
        return _to_version_out(v)


class ForkVersionIn(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    notes: str | None = None


@router.post(
    "/{dataset_id}/versions/{version_id}/fork",
    response_model=DatasetVersionOut,
    status_code=201,
)
def fork_version(
    dataset_id: str,
    version_id: str,
    body: ForkVersionIn,
    user: CurrentUser = Depends(require_user),
) -> DatasetVersionOut:
    """Create a new ``DatasetVersion`` by copying every sample from a parent.

    Audio is not duplicated: ``AudioRef.uri`` is preserved by reference. Only
    ``manifest.json`` and ``samples.jsonl`` are written under the new version
    directory. The new row's ``parent_version_id`` records the provenance link.
    """
    settings = get_settings()
    with session_scope() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404, "dataset not found")
        assert_role(user, d.project_id, Role.EDITOR)
        parent = s.get(DatasetVersion, version_id)
        if not parent:
            raise HTTPException(404, "parent version not found")
        # Cross-dataset forks are rejected: provenance must stay within the
        # same logical dataset so that lineage traversals remain meaningful.
        if parent.dataset_id != dataset_id:
            raise HTTPException(400, "parent version belongs to a different dataset")
        parent_root = Path(parent.manifest_uri.removeprefix("file://"))

        new_root = settings.datasets_dir / dataset_id / body.version
        if new_root.exists() and any(new_root.iterdir()):
            raise HTTPException(409, f"version {body.version!r} already exists on disk")

        # Inherit license_default from the parent header so the fork stays
        # licence-consistent without forcing the caller to re-specify it.
        parent_reader = ManifestReader(parent_root)
        header = ManifestHeader(
            dataset_id=dataset_id,
            dataset_version=body.version,
            name=f"{d.slug}:{body.version}",
            modality=d.modality,
            license_default=parent_reader.header.license_default,
        )
        init_manifest(new_root, header)
        for sample in ManifestReader(parent_root):
            manifest_append_sample(new_root, sample)

        reader = ManifestReader(new_root)
        v = DatasetVersion(
            dataset_id=dataset_id,
            version=body.version,
            manifest_uri=f"file://{new_root}",
            num_samples=reader.header.stats.num_samples,
            total_audio_s=reader.header.stats.total_audio_s,
            stats=reader.header.stats.model_dump(mode="json"),
            notes=body.notes,
            parent_version_id=parent.id,
        )
        s.add(v)
        s.flush()
        return _to_version_out(v)


# Discriminated union mirrors the manifest Sample type.
_SAMPLE_ADAPTER: TypeAdapter[ASRSample | TTSSample | LLMSample | S2SSample] = TypeAdapter(
    ASRSample | TTSSample | LLMSample | S2SSample,
)


class AppendSampleIn(BaseModel):
    sample: dict[str, Any]


@router.post("/versions/{version_id}/samples", status_code=201)
def append_sample_endpoint(
    version_id: str,
    body: AppendSampleIn,
    user: CurrentUser = Depends(require_user),
) -> dict[str, Any]:
    """Append a single typed sample to an existing manifest version.

    Used by the in-UI dataset builders. The sample must validate against the
    manifest Sample discriminated union (modality field is the discriminator).
    """
    with session_scope() as s:
        v = s.get(DatasetVersion, version_id)
        if not v:
            raise HTTPException(404)
        d = s.get(Dataset, v.dataset_id)
        if d:
            assert_role(user, d.project_id, Role.EDITOR)
        root = Path(v.manifest_uri.removeprefix("file://"))

    try:
        sample = _SAMPLE_ADAPTER.validate_python(body.sample)
    except Exception as e:
        raise HTTPException(400, f"invalid sample: {e}") from e
    manifest_append_sample(root, sample)

    # Refresh the DB count so listings stay accurate.
    reader = ManifestReader(root)
    with session_scope() as s:
        v2 = s.get(DatasetVersion, version_id)
        if v2:
            v2.num_samples = reader.header.stats.num_samples
            v2.total_audio_s = reader.header.stats.total_audio_s
            v2.stats = reader.header.stats.model_dump(mode="json")

    return {"id": sample.id, "num_samples": reader.header.stats.num_samples}


class AssignSplitsIn(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    train_pct: float = Field(default=0.8, ge=0.0, le=1.0)
    val_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    test_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    holdout_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: Literal["random", "speaker_disjoint"] = "random"
    notes: str | None = None
    seed: int = 42


@router.post(
    "/{dataset_id}/versions/{version_id}/split",
    response_model=DatasetVersionOut,
    status_code=201,
)
def assign_splits(
    dataset_id: str,
    version_id: str,
    body: AssignSplitsIn,
    user: CurrentUser = Depends(require_user),
) -> DatasetVersionOut:
    """Partition a dataset version's samples into splits (train/val/test/holdout).

    Writes a new forked version with split assignments updated for each sample.
    """
    total_pct = body.train_pct + body.val_pct + body.test_pct + body.holdout_pct
    if abs(total_pct - 1.0) > 1e-4:
        raise HTTPException(400, "train_pct + val_pct + test_pct + holdout_pct must sum to 1.0")

    settings = get_settings()
    with session_scope() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404, "dataset not found")
        assert_role(user, d.project_id, Role.EDITOR)
        parent = s.get(DatasetVersion, version_id)
        if not parent:
            raise HTTPException(404, "parent version not found")
        if parent.dataset_id != dataset_id:
            raise HTTPException(400, "parent version belongs to a different dataset")
        parent_root = Path(parent.manifest_uri.removeprefix("file://"))

        new_root = settings.datasets_dir / dataset_id / body.version
        if new_root.exists() and any(new_root.iterdir()):
            raise HTTPException(409, f"version {body.version!r} already exists on disk")

        # Load samples to memory to perform splitting/sorting.
        parent_reader = ManifestReader(parent_root)
        samples = list(parent_reader)
        N = len(samples)

        if N > 0:
            import random
            rng = random.Random(body.seed)

            if body.strategy == "random":
                rng.shuffle(samples)
                train_end = int(round(N * body.train_pct))
                val_end = int(round(N * (body.train_pct + body.val_pct)))
                test_end = int(round(N * (body.train_pct + body.val_pct + body.test_pct)))

                for i, sample in enumerate(samples):
                    if i < train_end:
                        sample.split = Split.TRAIN
                    elif i < val_end:
                        sample.split = Split.VAL
                    elif i < test_end:
                        sample.split = Split.TEST
                    else:
                        sample.split = Split.HOLDOUT

            elif body.strategy == "speaker_disjoint":
                speaker_groups: dict[str, list[Any]] = {}
                for idx, sample in enumerate(samples):
                    spk = getattr(sample, "speaker_id", None) or f"__unknown_{idx}"
                    speaker_groups.setdefault(spk, []).append(sample)

                spk_ids = list(speaker_groups.keys())
                rng.shuffle(spk_ids)

                target_train = N * body.train_pct
                target_val = N * body.val_pct
                target_test = N * body.test_pct

                total_assigned = 0
                for spk in spk_ids:
                    group_samples = speaker_groups[spk]
                    group_size = len(group_samples)
                    
                    if total_assigned < target_train:
                        split_val = Split.TRAIN
                    elif total_assigned < target_train + target_val:
                        split_val = Split.VAL
                    elif total_assigned < target_train + target_val + target_test:
                        split_val = Split.TEST
                    else:
                        split_val = Split.HOLDOUT
                    
                    for sample in group_samples:
                        sample.split = split_val
                    total_assigned += group_size

        # Write partitioned samples using ManifestWriter
        header = ManifestHeader(
            dataset_id=dataset_id,
            dataset_version=body.version,
            name=f"{d.slug}:{body.version}",
            modality=d.modality,
            license_default=parent_reader.header.license_default,
        )
        with ManifestWriter(new_root, header) as writer:
            writer.extend(samples)

        stats = writer.header.stats

        v = DatasetVersion(
            dataset_id=dataset_id,
            version=body.version,
            manifest_uri=f"file://{new_root}",
            num_samples=stats.num_samples,
            total_audio_s=stats.total_audio_s,
            stats=stats.model_dump(mode="json"),
            notes=body.notes,
            parent_version_id=parent.id,
        )
        s.add(v)
        s.flush()
        return _to_version_out(v)


class QualityFilterIn(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    min_duration_s: float | None = Field(default=None, ge=0.0)
    max_duration_s: float | None = Field(default=None, ge=0.0)
    min_snr_db: float | None = Field(default=None)
    min_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    min_text_len: int | None = Field(default=None, ge=0)
    max_text_len: int | None = Field(default=None, ge=0)
    notes: str | None = None


@router.post(
    "/{dataset_id}/versions/{version_id}/filter",
    response_model=DatasetVersionOut,
    status_code=201,
)
def quality_filter_version(
    dataset_id: str,
    version_id: str,
    body: QualityFilterIn,
    user: CurrentUser = Depends(require_user),
) -> DatasetVersionOut:
    """Filter a dataset version's samples based on quality metric thresholds.

    Writes a new forked version containing only the matching samples.
    """
    if body.min_duration_s is not None and body.max_duration_s is not None:
        if body.min_duration_s > body.max_duration_s:
            raise HTTPException(400, "min_duration_s must be <= max_duration_s")
    if body.min_text_len is not None and body.max_text_len is not None:
        if body.min_text_len > body.max_text_len:
            raise HTTPException(400, "min_text_len must be <= max_text_len")

    settings = get_settings()
    with session_scope() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404, "dataset not found")
        assert_role(user, d.project_id, Role.EDITOR)
        parent = s.get(DatasetVersion, version_id)
        if not parent:
            raise HTTPException(404, "parent version not found")
        if parent.dataset_id != dataset_id:
            raise HTTPException(400, "parent version belongs to a different dataset")
        parent_root = Path(parent.manifest_uri.removeprefix("file://"))

        new_root = settings.datasets_dir / dataset_id / body.version
        if new_root.exists() and any(new_root.iterdir()):
            raise HTTPException(409, f"version {body.version!r} already exists on disk")

        parent_reader = ManifestReader(parent_root)
        filtered_samples = []

        for sample in parent_reader:
            # Min Duration filter
            if body.min_duration_s is not None:
                audio = getattr(sample, "audio", None)
                if audio is None or audio.duration_s < body.min_duration_s:
                    continue
            
            # Max Duration filter
            if body.max_duration_s is not None:
                audio = getattr(sample, "audio", None)
                if audio is None or audio.duration_s > body.max_duration_s:
                    continue

            # SNR filter
            if body.min_snr_db is not None:
                audio = getattr(sample, "audio", None)
                if audio is None or audio.snr_db is None or audio.snr_db < body.min_snr_db:
                    continue

            # Quality Score filter
            if body.min_quality_score is not None:
                q = getattr(sample, "quality_score", None)
                if q is None or q < body.min_quality_score:
                    continue

            # Text Length filters
            if body.min_text_len is not None or body.max_text_len is not None:
                text = getattr(sample, "transcript", None) or getattr(sample, "text", None)
                if text is None:
                    continue
                t_len = len(text)
                if body.min_text_len is not None and t_len < body.min_text_len:
                    continue
                if body.max_text_len is not None and t_len > body.max_text_len:
                    continue

            filtered_samples.append(sample)

        # Write matching samples using ManifestWriter
        header = ManifestHeader(
            dataset_id=dataset_id,
            dataset_version=body.version,
            name=f"{d.slug}:{body.version}",
            modality=d.modality,
            license_default=parent_reader.header.license_default,
        )
        with ManifestWriter(new_root, header) as writer:
            writer.extend(filtered_samples)

        stats = writer.header.stats

        v = DatasetVersion(
            dataset_id=dataset_id,
            version=body.version,
            manifest_uri=f"file://{new_root}",
            num_samples=stats.num_samples,
            total_audio_s=stats.total_audio_s,
            stats=stats.model_dump(mode="json"),
            notes=body.notes,
            parent_version_id=parent.id,
        )
        s.add(v)
        s.flush()
        return _to_version_out(v)


class DedupIn(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    strategy: Literal["exact_text", "audio_hash", "similar_text"] = "exact_text"
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    notes: str | None = None


@router.post(
    "/{dataset_id}/versions/{version_id}/dedup",
    response_model=DatasetVersionOut,
    status_code=201,
)
def deduplicate_version(
    dataset_id: str,
    version_id: str,
    body: DedupIn,
    user: CurrentUser = Depends(require_user),
) -> DatasetVersionOut:
    """Run deduplication on a dataset version.

    Writes a new forked version containing only the unique samples.
    """
    settings = get_settings()
    with session_scope() as s:
        d = s.get(Dataset, dataset_id)
        if not d:
            raise HTTPException(404, "dataset not found")
        assert_role(user, d.project_id, Role.EDITOR)
        parent = s.get(DatasetVersion, version_id)
        if not parent:
            raise HTTPException(404, "parent version not found")
        if parent.dataset_id != dataset_id:
            raise HTTPException(400, "parent version belongs to a different dataset")
        parent_root = Path(parent.manifest_uri.removeprefix("file://"))

        new_root = settings.datasets_dir / dataset_id / body.version
        if new_root.exists() and any(new_root.iterdir()):
            raise HTTPException(409, f"version {body.version!r} already exists on disk")

        parent_reader = ManifestReader(parent_root)
        unique_samples = []

        if body.strategy == "exact_text":
            seen_texts = set()
            for sample in parent_reader:
                text = getattr(sample, "transcript", None) or getattr(sample, "text", None)
                if text is not None:
                    norm = text.strip().lower()
                    if norm in seen_texts:
                        continue
                    seen_texts.add(norm)
                unique_samples.append(sample)

        elif body.strategy == "audio_hash":
            seen_hashes = set()
            for sample in parent_reader:
                audio = getattr(sample, "audio", None)
                if audio is not None and audio.sha256 is not None:
                    h = audio.sha256
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                unique_samples.append(sample)

        elif body.strategy == "similar_text":
            import re
            seen_word_sets = []
            for sample in parent_reader:
                text = getattr(sample, "transcript", None) or getattr(sample, "text", None)
                if text is not None:
                    words = set(re.findall(r"\w+", text.lower()))
                    if words:
                        is_duplicate = False
                        for seen in seen_word_sets:
                            intersection = len(words & seen)
                            union = len(words | seen)
                            jaccard = intersection / union if union > 0 else 0.0
                            if jaccard >= body.threshold:
                                is_duplicate = True
                                break
                        if is_duplicate:
                            continue
                        seen_word_sets.append(words)
                unique_samples.append(sample)

        # Write matching samples using ManifestWriter
        header = ManifestHeader(
            dataset_id=dataset_id,
            dataset_version=body.version,
            name=f"{d.slug}:{body.version}",
            modality=d.modality,
            license_default=parent_reader.header.license_default,
        )
        with ManifestWriter(new_root, header) as writer:
            writer.extend(unique_samples)

        stats = writer.header.stats

        v = DatasetVersion(
            dataset_id=dataset_id,
            version=body.version,
            manifest_uri=f"file://{new_root}",
            num_samples=stats.num_samples,
            total_audio_s=stats.total_audio_s,
            stats=stats.model_dump(mode="json"),
            notes=body.notes,
            parent_version_id=parent.id,
        )
        s.add(v)
        s.flush()
        return _to_version_out(v)



