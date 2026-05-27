"""Lazy model loaders keyed by ModelVersion.id."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from oas_core.db import ModelVersion, session_scope


def _resolve(version_id: str) -> tuple[str, str]:
    """Return (artifact_path-or-hf-id, format) for a ModelVersion."""
    with session_scope() as s:
        v = s.get(ModelVersion, version_id)
        if not v:
            raise KeyError(f"model version {version_id!r} not found")
        target = v.artifact_uri.removeprefix("file://").removeprefix("hf://")
        return target, v.format


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


class ASRServer:
    def __init__(self, target: str) -> None:
        self._fw = None
        self._hf = None
        try:
            from faster_whisper import WhisperModel

            size = target.split("/")[-1].replace("whisper-", "")
            self._fw = WhisperModel(size, compute_type="int8")
        except Exception:
            from transformers import pipeline

            self._hf = pipeline("automatic-speech-recognition", model=target)

    def transcribe(self, pcm_f32: Any, sr: int, language: str | None = None) -> dict[str, Any]:
        if self._fw is not None:
            segments, info = self._fw.transcribe(pcm_f32, language=language, beam_size=1)
            text_parts = []
            words = []
            for seg in segments:
                text_parts.append(seg.text.strip())
                for w in getattr(seg, "words", []) or []:
                    words.append({"start": w.start, "end": w.end, "word": w.word})
            return {
                "text": " ".join(text_parts).strip(),
                "language": getattr(info, "language", language),
                "words": words,
            }
        assert self._hf is not None
        out = self._hf({"array": pcm_f32, "sampling_rate": sr})
        return {"text": str(out.get("text", "")).strip(), "language": language, "words": []}


def load_asr(version_id: str) -> ASRServer:
    target, _ = _resolve(version_id)
    return ASRServer(target)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class LLMServer:
    def __init__(self, target: str, fmt: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # PEFT adapter format: requires a base model. For now require the
        # caller to pass it; defer to direct AutoModel for non-adapter formats.
        if fmt == "peft-adapter":
            raise ValueError(
                "peft-adapter format requires a base_model — use load_llm_with_base()"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(target, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=dtype)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def stream(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Any:
        from threading import Thread

        from transformers import TextIteratorStreamer

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        Thread(target=self.model.generate, kwargs=gen_kwargs).start()
        for piece in streamer:
            if piece:
                yield piece


def load_llm(version_id: str, *, base_model: str | None = None) -> LLMServer:
    target, fmt = _resolve(version_id)
    if fmt == "peft-adapter":
        if not base_model:
            raise ValueError("peft-adapter format requires base_model")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        srv = LLMServer.__new__(LLMServer)
        srv.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if srv.tokenizer.pad_token is None:
            srv.tokenizer.pad_token = srv.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        srv.model = PeftModel.from_pretrained(base, target)
        srv.device = "cuda" if torch.cuda.is_available() else "cpu"
        srv.model.to(srv.device)
        srv.model.eval()
        return srv
    return LLMServer(target, fmt)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TTSServer:
    def __init__(self, target: str, fmt: str) -> None:
        self._piper = None
        self._hf = None
        if fmt == "piper-onnx":
            from piper import PiperVoice

            onnx = next(Path(target).glob("*.onnx"))
            self._piper = PiperVoice.load(str(onnx))
        else:
            from transformers import pipeline

            self._hf = pipeline("text-to-speech", model=target)

    def synth(self, text: str) -> tuple[Any, int]:
        import numpy as np

        if self._piper is not None:
            buf = io.BytesIO()
            self._piper.synthesize(text, buf)
            buf.seek(0)
            import soundfile as sf

            data, sr = sf.read(buf, dtype="float32", always_2d=False)
            return data.astype(np.float32), int(sr)
        assert self._hf is not None
        out = self._hf(text)
        return (
            __import__("numpy").asarray(out["audio"], dtype="float32").squeeze(),
            int(out["sampling_rate"]),
        )


def load_tts(version_id: str) -> TTSServer:
    target, fmt = _resolve(version_id)
    return TTSServer(target, fmt)
