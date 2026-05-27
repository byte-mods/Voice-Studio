"""From-scratch pretraining handlers.

Each handler takes:
- `model_spec_id`: ModelSpec row whose `spec` is fed to the architecture
  factory to build a fresh `nn.Module` (no pretrained weights).
- `dataset_version_id`: manifest version supplying training data.
- A small training-config dict (epochs / batch / lr / etc.).

All three handlers use the studio's existing training plumbing:
- HF `Trainer` for batching and metric logging hooks.
- `JobContext.log` + the `step k=v` log format so the live-charts UI lights up.
- `publish_version()` on success.

For v1 the tokenizer is a tiny byte-level BPE (LLM) or a fixed character
vocab (ASR / TTS) so the studio can run end-to-end without a precomputed
tokenizer. Real workflows will overlay a proper tokenizer via plugin.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from oas_core.architectures import build_from_spec, validate_spec
from oas_core.db import DatasetVersion, ModelSpec, session_scope
from oas_core.manifest import (
    ASRSample,
    LLMSample,
    ManifestReader,
    Modality,
    TTSSample,
)
from oas_core.queue.backend import JobContext, register_handler
from oas_core.registry import publish_version

log = logging.getLogger(__name__)


def _load_spec(spec_id: str) -> tuple[dict[str, Any], str]:
    with session_scope() as s:
        ms = s.get(ModelSpec, spec_id)
        if not ms:
            raise ValueError(f"ModelSpec {spec_id!r} not found")
        spec = dict(ms.spec or {})
        project_id = ms.project_id
        return spec, project_id


def _load_manifest_root(dv_id: str) -> Path:
    with session_scope() as s:
        v = s.get(DatasetVersion, dv_id)
        if not v:
            raise ValueError(f"DatasetVersion {dv_id!r} not found")
        return Path(v.manifest_uri.removeprefix("file://"))


def _common_save_and_publish(
    ctx: JobContext,
    spec: dict[str, Any],
    project_id: str,
    out_dir: Path,
    metrics: dict[str, Any],
    registry_cfg: dict[str, Any] | None,
    dv_id: str,
    fmt: str,
) -> dict[str, Any]:
    if registry_cfg and registry_cfg.get("model_id") and registry_cfg.get("version"):
        try:
            publish_version(
                model_id=registry_cfg["model_id"],
                version=registry_cfg["version"],
                artifact_uri=f"file://{out_dir}",
                format=fmt,
                metrics=metrics,
                source_run_id=ctx.run_id,
                source_dataset_version_id=dv_id,
                notes=f"from-scratch pretrain: {spec.get('modality')} "
                f"h={spec.get('hidden_size')} L={spec.get('num_layers')}",
            )
            ctx.log(f"published version {registry_cfg['version']}")
        except Exception as e:
            ctx.log(f"publish failed: {e}")
    return {"artifact_uri": f"file://{out_dir}", **metrics}


# ---------------------------------------------------------------------------
# LLM pretraining
# ---------------------------------------------------------------------------


class _ByteTokenizer:
    """Tiny tokenizer used when the user hasn't supplied one: each Unicode
    codepoint becomes a token id (mod vocab_size). Trivially reversible and
    good enough for proving the training loop works end-to-end."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]


def _llm_text_iter(manifest_root: Path, split: str) -> Any:
    for s in ManifestReader(manifest_root):
        if s.modality != Modality.LLM or s.split.value != split:
            continue
        ls: LLMSample = s
        parts = []
        if ls.system_prompt:
            parts.append(f"<|system|>\n{ls.system_prompt}")
        for t in ls.turns:
            parts.append(f"<|{t.role.value}|>\n{t.text or ''}")
        yield "\n".join(parts)


def llm_pretrain_handler(ctx: JobContext) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    cfg = ctx.config
    spec_id = cfg["model_spec_id"]
    dv_id = cfg["dataset_version_id"]
    training_cfg = cfg.get("training", {})
    registry_cfg = cfg.get("registry")

    epochs = int(training_cfg.get("epochs", 1))
    batch_size = int(training_cfg.get("batch_size", 4))
    seq_len = int(training_cfg.get("max_seq_len", 512))
    lr = float(training_cfg.get("learning_rate", 3e-4))

    spec, project_id = _load_spec(spec_id)
    validate_spec(spec)
    if spec.get("modality") != "llm":
        raise ValueError("llm_pretrain requires an LLM spec")
    root = _load_manifest_root(dv_id)

    ctx.log(f"building LLM from spec: {spec}")
    model = build_from_spec(spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tok = _ByteTokenizer(spec["vocab_size"])

    class _DS(IterableDataset):  # type: ignore[misc]
        def __iter__(self) -> Any:
            for text in _llm_text_iter(root, "train"):
                ids = tok.encode(text)[: seq_len + 1]
                if len(ids) < 2:
                    continue
                ids = ids + [0] * (seq_len + 1 - len(ids))
                t = torch.tensor(ids, dtype=torch.long)
                yield {"input_ids": t[:-1], "labels": t[1:]}

    loader = DataLoader(_DS(), batch_size=batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    last_loss = float("nan")
    for epoch in range(epochs):
        for batch in loader:
            if ctx.cancelled:
                ctx.log("cancelled")
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            if step % 10 == 0:
                ctx.log(f"step loss={last_loss:.4f} epoch={epoch}")
                ctx.heartbeat()
            step += 1

    out_dir = Path(ctx.artifacts_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "spec": spec}, out_dir / "model.pt")
    ctx.log(f"saved checkpoint to {out_dir}")
    return _common_save_and_publish(
        ctx, spec, project_id, out_dir, {"final_loss": last_loss, "steps": step},
        registry_cfg, dv_id, fmt="from-scratch-pt",
    )


# ---------------------------------------------------------------------------
# ASR pretraining
# ---------------------------------------------------------------------------


def _mel_features(audio: Any, sr: int, n_mels: int) -> Any:
    import numpy as np
    import torch

    try:
        import librosa

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=160, n_fft=400)
        log_mel = np.log(mel + 1e-9).astype("float32")
        return torch.from_numpy(log_mel.T)  # (T, n_mels)
    except ImportError:
        # Fallback: just bin the waveform into n_mels averages — placeholder
        # so the training loop still runs without librosa.
        x = torch.from_numpy(audio).float()
        chunks = max(1, x.numel() // n_mels)
        x = x[: chunks * n_mels].view(chunks, n_mels)
        return x


def asr_pretrain_handler(ctx: JobContext) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    cfg = ctx.config
    spec_id = cfg["model_spec_id"]
    dv_id = cfg["dataset_version_id"]
    training_cfg = cfg.get("training", {})
    registry_cfg = cfg.get("registry")

    epochs = int(training_cfg.get("epochs", 1))
    batch_size = int(training_cfg.get("batch_size", 2))
    lr = float(training_cfg.get("learning_rate", 1e-4))

    spec, project_id = _load_spec(spec_id)
    validate_spec(spec)
    if spec.get("modality") != "asr":
        raise ValueError("asr_pretrain requires an ASR spec")
    n_mels = (spec.get("encoder") or {}).get("n_mels", 80)
    vocab = spec.get("vocab_size", 256)
    root = _load_manifest_root(dv_id)

    ctx.log(f"building ASR from spec: {spec}")
    model = build_from_spec(spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def char_encode(text: str) -> list[int]:
        return [max(1, ord(c) % vocab) for c in text]

    class _DS(IterableDataset):  # type: ignore[misc]
        def __iter__(self) -> Any:
            for s in ManifestReader(root):
                if s.modality != Modality.ASR or s.split.value != "train":
                    continue
                samp: ASRSample = s
                path = Path(samp.audio.uri.removeprefix("file://"))
                try:
                    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
                except Exception:
                    continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                mel = _mel_features(audio, sr, n_mels)
                tgt = torch.tensor(char_encode(samp.transcript or ""), dtype=torch.long)
                if tgt.numel() == 0:
                    continue
                yield {"mel": mel, "targets": tgt}

    def _collate(batch: Any) -> Any:
        from torch.nn.utils.rnn import pad_sequence

        mels = pad_sequence([b["mel"] for b in batch], batch_first=True)
        targets = pad_sequence([b["targets"] for b in batch], batch_first=True, padding_value=0)
        return {"mel": mels, "targets": targets}

    loader = DataLoader(_DS(), batch_size=batch_size, collate_fn=_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    last_loss = float("nan")
    for epoch in range(epochs):
        for batch in loader:
            if ctx.cancelled:
                break
            mel = batch["mel"].to(device)
            targets = batch["targets"].to(device)
            out = model(mel, targets=targets)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            if step % 5 == 0:
                ctx.log(f"step loss={last_loss:.4f} epoch={epoch}")
                ctx.heartbeat()
            step += 1

    out_dir = Path(ctx.artifacts_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "spec": spec}, out_dir / "model.pt")
    return _common_save_and_publish(
        ctx, spec, project_id, out_dir, {"final_ctc_loss": last_loss, "steps": step},
        registry_cfg, dv_id, fmt="from-scratch-pt",
    )


# ---------------------------------------------------------------------------
# TTS pretraining
# ---------------------------------------------------------------------------


def tts_pretrain_handler(ctx: JobContext) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    cfg = ctx.config
    spec_id = cfg["model_spec_id"]
    dv_id = cfg["dataset_version_id"]
    training_cfg = cfg.get("training", {})
    registry_cfg = cfg.get("registry")

    epochs = int(training_cfg.get("epochs", 1))
    batch_size = int(training_cfg.get("batch_size", 2))
    lr = float(training_cfg.get("learning_rate", 3e-4))

    spec, project_id = _load_spec(spec_id)
    validate_spec(spec)
    if spec.get("modality") != "tts":
        raise ValueError("tts_pretrain requires a TTS spec")
    n_mels = (spec.get("tts") or {}).get("n_mels", 80)
    vocab = spec.get("vocab_size", 256)
    root = _load_manifest_root(dv_id)

    ctx.log(f"building TTS from spec: {spec}")
    model = build_from_spec(spec)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def char_encode(text: str) -> list[int]:
        return [max(1, ord(c) % vocab) for c in text]

    class _DS(IterableDataset):  # type: ignore[misc]
        def __iter__(self) -> Any:
            for s in ManifestReader(root):
                if s.modality != Modality.TTS or s.split.value != "train":
                    continue
                samp: TTSSample = s
                path = Path(samp.audio.uri.removeprefix("file://"))
                try:
                    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
                except Exception:
                    continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                mel = _mel_features(audio, sr, n_mels)
                text_ids = torch.tensor(char_encode(samp.text or ""), dtype=torch.long)
                if text_ids.numel() == 0:
                    continue
                yield {"text_ids": text_ids, "mel": mel}

    def _collate(batch: Any) -> Any:
        from torch.nn.utils.rnn import pad_sequence

        text_ids = pad_sequence([b["text_ids"] for b in batch], batch_first=True, padding_value=0)
        mel = pad_sequence([b["mel"] for b in batch], batch_first=True)
        return {"text_ids": text_ids, "mel": mel}

    loader = DataLoader(_DS(), batch_size=batch_size, collate_fn=_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    last_loss = float("nan")
    for epoch in range(epochs):
        for batch in loader:
            if ctx.cancelled:
                break
            text_ids = batch["text_ids"].to(device)
            mel = batch["mel"].to(device)
            out = model(text_ids, mel_target=mel)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            if step % 5 == 0:
                ctx.log(f"step loss={last_loss:.4f} epoch={epoch}")
                ctx.heartbeat()
            step += 1

    out_dir = Path(ctx.artifacts_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "spec": spec}, out_dir / "model.pt")
    return _common_save_and_publish(
        ctx, spec, project_id, out_dir, {"final_l1_loss": last_loss, "steps": step},
        registry_cfg, dv_id, fmt="from-scratch-pt",
    )


for _kind, _fn in (
    ("llm_pretrain", llm_pretrain_handler),
    ("asr_pretrain", asr_pretrain_handler),
    ("tts_pretrain", tts_pretrain_handler),
):
    with contextlib.suppress(ValueError):
        register_handler(_kind, _fn)
