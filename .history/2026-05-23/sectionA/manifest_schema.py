"""Dataset manifest schema v1.

Design goals:
- One schema covers ASR, TTS, LLM, and S2S samples with shared metadata.
- Forward-compatible via `metadata` free-form dict and explicit `schema_version`.
- License + consent are first-class so the studio can enforce policy.
- Splits are speaker-disjoint by default for audio modalities.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Modality(StrEnum):
    ASR = "asr"
    TTS = "tts"
    LLM = "llm"
    S2S = "s2s"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HOLDOUT = "holdout"


class AudioCodec(StrEnum):
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    OPUS = "opus"
    MP3 = "mp3"
    M4A = "m4a"
    PCM_S16LE = "pcm_s16le"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------


class LicenseInfo(BaseModel):
    """Licensing info attached to every sample. The studio refuses to train on
    samples missing a license unless the project explicitly opts out."""

    model_config = ConfigDict(extra="forbid")

    spdx: str = Field(description="SPDX identifier, e.g. 'CC-BY-4.0', 'MIT', 'proprietary'.")
    holder: str | None = Field(default=None, description="Copyright holder / source org.")
    source_url: str | None = None
    notes: str | None = None


class ConsentRecord(BaseModel):
    """Voice / speaker consent record. Required for TTS voice cloning."""

    model_config = ConfigDict(extra="forbid")

    consent_id: str
    speaker_id: str
    granted_at: datetime
    expires_at: datetime | None = None
    scope: list[str] = Field(
        default_factory=list,
        description="e.g. ['tts_clone', 'asr_training', 'public_demo'].",
    )
    document_uri: str | None = Field(
        default=None, description="Pointer to the signed consent document."
    )


class AudioRef(BaseModel):
    """Reference to an audio artifact in storage."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(description="Storage URI, e.g. 'file://...', 's3://...'.")
    codec: AudioCodec = AudioCodec.WAV
    sample_rate: int = Field(ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=8)
    duration_s: float = Field(ge=0)
    sha256: str | None = Field(default=None, description="Content hash for dedup / integrity.")
    bit_depth: int | None = None
    loudness_lufs: float | None = None
    snr_db: float | None = None


class TimeSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)

    @field_validator("end_s")
    @classmethod
    def _end_after_start(cls, v: float, info: Any) -> float:
        start = info.data.get("start_s", 0.0)
        if v < start:
            raise ValueError("end_s must be >= start_s")
        return v


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    start_s: float
    end_s: float
    confidence: float | None = None
    speaker_id: str | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_call_id: str
    content: Any
    is_error: bool = False


class DialogTurn(BaseModel):
    """One conversational turn. Used by LLM and S2S samples."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    text: str | None = None
    audio: AudioRef | None = None
    speaker_id: str | None = None
    started_at_s: float | None = Field(
        default=None, description="Offset from conversation start, for S2S alignment."
    )
    ended_at_s: float | None = None
    interrupted: bool = False
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sample types
# ---------------------------------------------------------------------------


class SampleBase(BaseModel):
    """Common fields on every sample, regardless of modality."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    modality: Modality
    split: Split = Split.TRAIN
    language: str | None = Field(default=None, description="BCP-47 tag, e.g. 'en', 'en-IN'.")
    license: LicenseInfo
    consent: ConsentRecord | None = None
    domain: str | None = Field(default=None, description="e.g. 'medical', 'finance', 'casual'.")
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str | None = Field(default=None, description="Provenance, e.g. HF dataset id.")
    quality_score: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ASRSample(SampleBase):
    modality: Literal[Modality.ASR] = Modality.ASR
    audio: AudioRef
    transcript: str
    speaker_id: str | None = None
    accent: str | None = None
    device: str | None = Field(default=None, description="e.g. 'iphone_15', 'studio_mic'.")
    noise_condition: str | None = None
    words: list[Word] = Field(default_factory=list, description="Optional word-level alignment.")


class TTSSample(SampleBase):
    modality: Literal[Modality.TTS] = Modality.TTS
    audio: AudioRef
    text: str
    speaker_id: str = Field(description="Required: TTS samples must have a speaker.")
    style: str | None = Field(default=None, description="e.g. 'neutral', 'cheerful'.")
    emotion: str | None = None
    speaking_rate: float | None = Field(default=None, gt=0)
    phonemes: str | None = None
    alignment_quality: float | None = Field(default=None, ge=0, le=1)


class LLMSample(SampleBase):
    modality: Literal[Modality.LLM] = Modality.LLM
    turns: list[DialogTurn]
    system_prompt: str | None = None
    tools_schema: list[dict[str, Any]] = Field(
        default_factory=list, description="JSON-Schema tool definitions available in this sample."
    )
    asr_noisified_from: str | None = Field(
        default=None,
        description="If this sample's user turns were derived from clean text via TTS→ASR.",
    )


class S2SSample(SampleBase):
    """Speech-to-speech conversation sample (Gemini Live-style)."""

    modality: Literal[Modality.S2S] = Modality.S2S
    turns: list[DialogTurn]
    user_audio: AudioRef | None = Field(
        default=None, description="Optional combined user-side audio track."
    )
    assistant_audio: AudioRef | None = Field(
        default=None, description="Optional combined assistant-side audio track."
    )
    interruptions: list[TimeSpan] = Field(default_factory=list)
    background_events: list[dict[str, Any]] = Field(default_factory=list)
    tools_schema: list[dict[str, Any]] = Field(default_factory=list)
    synthetic: bool = Field(
        default=False, description="True if generated via LLM→TTS→ASR synthesis."
    )


Sample = Annotated[
    ASRSample | TTSSample | LLMSample | S2SSample,
    Field(discriminator="modality"),
]


# ---------------------------------------------------------------------------
# Manifest container
# ---------------------------------------------------------------------------


class ManifestStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_samples: int = 0
    num_speakers: int = 0
    total_audio_s: float = 0.0
    by_split: dict[str, int] = Field(default_factory=dict)
    by_language: dict[str, int] = Field(default_factory=dict)


class ManifestHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    dataset_id: str
    dataset_version: str = "0.1.0"
    name: str
    modality: Modality
    description: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str | None = None
    source: str | None = None
    license_default: LicenseInfo | None = None
    stats: ManifestStats = Field(default_factory=ManifestStats)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Manifest(BaseModel):
    """In-memory manifest. For large datasets prefer `ManifestReader` / `ManifestWriter`
    which stream samples to/from disk without loading everything into RAM."""

    model_config = ConfigDict(extra="forbid")

    header: ManifestHeader
    samples: list[Sample] = Field(default_factory=list)
