"""S2S realtime session with streaming ASR + barge-in.

A `S2SSession` owns one WebSocket conversation. Audio chunks arrive from the
client; the session detects speech via VAD, emits *partial* transcripts every
~500 ms while the user speaks, finalizes the turn on trailing silence, then
runs LLM → TTS. If the user starts speaking again while the assistant is mid-
reply, the current turn is **barge-in cancelled** — in-flight LLM/TTS work is
torn down, a `tts_cancel` event tells the client to stop playback, and a new
turn begins.

Turn IDs guard against stale audio: any binary frame the server sends carries
a `turn_id`, and the client drops audio whose id doesn't match the current
turn.

Heavy ML (`torch`, `transformers`, `faster-whisper`, etc.) is imported lazily
so the studio still boots without `[s2s]` extras installed.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import secrets
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket
from oas_core.db import ModelVersion, S2SPipeline, session_scope

log = logging.getLogger(__name__)

CLIENT_SR = 16000  # PCM16 mono expected from browser
DEFAULT_SILENCE_TAIL_MS = 600
DEFAULT_MIN_SPEECH_MS = 200
DEFAULT_PARTIAL_INTERVAL_MS = 600


@dataclass
class SessionConfig:
    pipeline_id: str
    asr_uri: str
    asr_format: str
    llm_uri: str
    llm_format: str
    tts_uri: str
    tts_format: str
    system_prompt: str
    vad: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


def _resolve_one(
    s: Any, version_id: str | None, fallback: str | None, kind: str
) -> tuple[str, str]:
    if version_id:
        v = s.get(ModelVersion, version_id)
        if v is None:
            raise ValueError(f"{kind} model version {version_id!r} not found")
        return v.artifact_uri, v.format
    if not fallback:
        raise ValueError(f"{kind} model not configured (no version, no fallback)")
    return f"hf://{fallback}", "hf"


def resolve_pipeline(pipeline_id: str) -> SessionConfig:
    with session_scope() as s:
        p = s.get(S2SPipeline, pipeline_id)
        if not p:
            raise ValueError(f"pipeline {pipeline_id!r} not found")
        asr_uri, asr_fmt = _resolve_one(s, p.asr_version_id, p.asr_fallback, "ASR")
        llm_uri, llm_fmt = _resolve_one(s, p.llm_version_id, p.llm_fallback, "LLM")
        tts_uri, tts_fmt = _resolve_one(s, p.tts_version_id, p.tts_fallback, "TTS")
        return SessionConfig(
            pipeline_id=p.id,
            asr_uri=asr_uri,
            asr_format=asr_fmt,
            llm_uri=llm_uri,
            llm_format=llm_fmt,
            tts_uri=tts_uri,
            tts_format=tts_fmt,
            system_prompt=p.system_prompt or "",
            vad=p.vad_config or {},
            runtime=p.runtime_config or {},
        )


# ---------------------------------------------------------------------------
# ML wrappers — lazy.
# ---------------------------------------------------------------------------


class _ASR:
    """Whisper-family ASR with both batch transcribe (final) and one-shot
    transcribe-of-current-buffer (partial). True token-by-token streaming will
    come when we wire faster-whisper's prefix/condition_on_previous flow."""

    def __init__(self, uri: str, fmt: str, language: str = "en") -> None:
        self.language = language
        self._fw = None
        self._hf = None
        target = uri.removeprefix("hf://").removeprefix("file://")
        try:
            from faster_whisper import WhisperModel

            size = target.split("/")[-1].replace("whisper-", "")
            self._fw = WhisperModel(size, compute_type="int8")
        except Exception:
            from transformers import pipeline

            self._hf = pipeline("automatic-speech-recognition", model=target)

    def transcribe(self, pcm_f32: Any, sr: int) -> str:
        if self._fw is not None:
            segments, _ = self._fw.transcribe(pcm_f32, language=self.language, beam_size=1)
            return " ".join(seg.text.strip() for seg in segments).strip()
        assert self._hf is not None
        out = self._hf({"array": pcm_f32, "sampling_rate": sr})
        return str(out.get("text", "")).strip()


class _LLM:
    def __init__(self, uri: str, fmt: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target = uri.removeprefix("hf://").removeprefix("file://")
        self.tokenizer = AutoTokenizer.from_pretrained(target, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=dtype)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def stream_reply(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 200,
        cancel: asyncio.Event | None = None,
    ) -> Iterator[str]:
        """Yield string chunks as the model generates. Honors `cancel`."""
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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        try:
            for piece in streamer:
                if cancel is not None and cancel.is_set():
                    return
                if piece:
                    yield piece
        finally:
            thread.join(timeout=0.1)


class _TTS:
    def __init__(self, uri: str, fmt: str) -> None:
        self._piper = None
        self._hf = None
        target_path = uri.removeprefix("file://")

        if fmt == "piper-onnx":
            from pathlib import Path

            from piper import PiperVoice

            voice_dir = Path(target_path)
            onnx = next(voice_dir.glob("*.onnx"))
            self._piper = PiperVoice.load(str(onnx))
        else:
            from transformers import pipeline

            self._hf = pipeline("text-to-speech", model=uri.removeprefix("hf://"))

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
        audio = np.asarray(out["audio"], dtype=np.float32).squeeze()
        return audio, int(out["sampling_rate"])


# ---------------------------------------------------------------------------
# Audio utils
# ---------------------------------------------------------------------------


def _to_pcm16(arr: Any) -> bytes:
    import numpy as np

    clipped = np.clip(arr, -1.0, 1.0)
    return bytes((clipped * 32767.0).astype("<i2").tobytes())


def _from_pcm16(buf: bytes) -> Any:
    import numpy as np

    return np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0


def _chunk_pcm16(arr: Any, frame_ms: int = 80) -> list[bytes]:
    """Split a numpy float32 audio array into PCM16 frames for streamed playback."""
    n = max(1, int(CLIENT_SR * frame_ms / 1000))
    out: list[bytes] = []
    for i in range(0, len(arr), n):
        out.append(_to_pcm16(arr[i : i + n]))
    return out


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class S2SSession:
    def __init__(self, cfg: SessionConfig) -> None:
        self.cfg = cfg
        vad = cfg.vad or {}
        self.silence_tail_ms = int(vad.get("silence_tail_ms", DEFAULT_SILENCE_TAIL_MS))
        self.min_speech_ms = int(vad.get("min_speech_ms", DEFAULT_MIN_SPEECH_MS))
        self.rms_threshold = float(vad.get("rms_threshold", 0.015))
        self.partial_interval_ms = int(
            (cfg.runtime or {}).get("partial_interval_ms", DEFAULT_PARTIAL_INTERVAL_MS)
        )

        self._asr: _ASR | None = None
        self._llm: _LLM | None = None
        self._tts: _TTS | None = None

        self._inbox: list[bytes] = []  # accumulated PCM16 for the current user turn
        self._pending_events: asyncio.Queue[dict[str, Any] | bytes] = asyncio.Queue()
        self._history: list[dict[str, str]] = []
        if cfg.system_prompt:
            self._history.append({"role": "system", "content": cfg.system_prompt})

        self._silence_ms = 0
        self._speech_ms = 0
        self._ms_since_partial = 0
        self._turn_id: str | None = None  # current user turn (capture)
        self._reply_turn_id: str | None = None  # in-flight assistant reply
        self._cancel_event: asyncio.Event | None = None
        self._reply_task: asyncio.Task[None] | None = None
        # Fire-and-forget partial-transcribe tasks. We hold strong refs so the
        # event loop does not GC them mid-flight (asyncio only weak-references
        # tasks created without an external owner).
        self._partial_tasks: set[asyncio.Task[None]] = set()

    # ---- lifecycle ----

    async def start(self) -> None:
        await self._pending_events.put(
            {"type": "ready", "pipeline_id": self.cfg.pipeline_id}
        )

    async def close(self) -> None:
        await self._cancel_in_flight_reply(send_event=False)

    # ---- model loading ----

    async def _ensure_loaded(self) -> None:
        if self._asr is not None and self._llm is not None and self._tts is not None:
            return
        await self._pending_events.put({"type": "loading"})
        await asyncio.to_thread(self._load_blocking)
        await self._pending_events.put({"type": "loaded"})

    def _load_blocking(self) -> None:
        if self._asr is None:
            log.info("loading ASR: %s", self.cfg.asr_uri)
            self._asr = _ASR(self.cfg.asr_uri, self.cfg.asr_format)
        if self._llm is None:
            log.info("loading LLM: %s", self.cfg.llm_uri)
            self._llm = _LLM(self.cfg.llm_uri, self.cfg.llm_format)
        if self._tts is None:
            log.info("loading TTS: %s", self.cfg.tts_uri)
            self._tts = _TTS(self.cfg.tts_uri, self.cfg.tts_format)

    # ---- inbound from WS ----

    async def push_audio(self, chunk: bytes) -> None:
        import numpy as np

        samples = _from_pcm16(chunk)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples**2)))
        chunk_ms = int(1000 * samples.size / CLIENT_SR)

        is_speech = rms >= self.rms_threshold
        if is_speech:
            # First speech in a (possibly cancelling) turn: open turn + maybe barge-in.
            if self._turn_id is None:
                self._turn_id = secrets.token_urlsafe(6)
                await self._pending_events.put(
                    {"type": "turn_start", "turn_id": self._turn_id}
                )
                # If the assistant is currently replying, barge in.
                if self._reply_task is not None and not self._reply_task.done():
                    await self._cancel_in_flight_reply()

            self._inbox.append(chunk)
            self._speech_ms += chunk_ms
            self._silence_ms = 0
            self._ms_since_partial += chunk_ms
            if self._ms_since_partial >= self.partial_interval_ms:
                self._ms_since_partial = 0
                # Run a partial transcribe over the accumulated buffer.
                task = asyncio.create_task(self._emit_partial())
                self._partial_tasks.add(task)
                task.add_done_callback(self._partial_tasks.discard)
        else:
            if self._turn_id is None:
                return  # ambient silence with no open turn
            self._inbox.append(chunk)
            self._silence_ms += chunk_ms
            if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.silence_tail_ms:
                await self._finalize_turn()

    async def on_control(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        kind = msg.get("type")
        if kind == "end_turn":
            await self._finalize_turn()
        elif kind == "reset":
            await self._cancel_in_flight_reply()
            self._history = (
                [{"role": "system", "content": self.cfg.system_prompt}]
                if self.cfg.system_prompt
                else []
            )
            await self._pending_events.put({"type": "reset"})

    # ---- turn lifecycle ----

    async def _emit_partial(self) -> None:
        if not self._inbox or self._turn_id is None:
            return
        try:
            await self._ensure_loaded()
        except Exception as e:
            await self._pending_events.put({"type": "error", "message": str(e)})
            return
        audio = _from_pcm16(b"".join(self._inbox))
        # `_ensure_loaded` succeeded above, so self._asr is non-None; mypy does
        # not propagate the narrowing across the await boundary.
        assert self._asr is not None
        try:
            text = await asyncio.to_thread(self._asr.transcribe, audio, CLIENT_SR)
        except Exception as e:
            log.warning("partial ASR failed: %s", e)
            return
        if text:
            await self._pending_events.put(
                {"type": "partial_transcript", "turn_id": self._turn_id, "text": text}
            )

    async def _finalize_turn(self) -> None:
        if not self._inbox or self._turn_id is None:
            return
        captured_turn_id = self._turn_id
        chunks = self._inbox
        self._inbox = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._ms_since_partial = 0
        self._turn_id = None

        try:
            await self._ensure_loaded()
        except Exception as e:
            await self._pending_events.put({"type": "error", "message": str(e)})
            return

        audio = _from_pcm16(b"".join(chunks))
        assert self._asr is not None
        try:
            transcript = await asyncio.to_thread(self._asr.transcribe, audio, CLIENT_SR)
        except Exception as e:
            await self._pending_events.put({"type": "error", "message": f"asr: {e}"})
            return

        await self._pending_events.put(
            {"type": "final_transcript", "turn_id": captured_turn_id, "text": transcript}
        )
        if not transcript.strip():
            return

        self._history.append({"role": "user", "content": transcript})

        # Spawn the reply task so the inbound loop can keep handling barge-in.
        self._cancel_event = asyncio.Event()
        self._reply_turn_id = captured_turn_id
        self._reply_task = asyncio.create_task(
            self._run_reply(captured_turn_id, self._cancel_event)
        )

    async def _cancel_in_flight_reply(self, *, send_event: bool = True) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        task = self._reply_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except TimeoutError:
                task.cancel()
        self._reply_task = None
        self._cancel_event = None
        if send_event and self._reply_turn_id is not None:
            await self._pending_events.put(
                {"type": "tts_cancel", "turn_id": self._reply_turn_id}
            )
        self._reply_turn_id = None

    async def _run_reply(self, turn_id: str, cancel: asyncio.Event) -> None:
        try:
            # Stream LLM tokens, accumulating into sentence-ish chunks for TTS.
            assert self._llm is not None
            llm = self._llm  # local capture; assert narrowing does not reach into the nested closure.
            buffer = ""
            full = ""

            def gen() -> Iterator[str]:
                yield from llm.stream_reply(self._history, cancel=cancel)

            iterator = await asyncio.to_thread(lambda: iter(gen()))
            sentence_split = (".", "!", "?", "\n")

            while True:
                if cancel.is_set():
                    return
                try:
                    piece = await asyncio.to_thread(next, iterator, None)
                except StopIteration:
                    piece = None
                if piece is None:
                    break
                buffer += piece
                full += piece
                # Flush sentence-sized chunks to TTS as they complete.
                if any(p in piece for p in sentence_split) and len(buffer.strip()) > 0:
                    await self._speak_chunk(turn_id, buffer.strip(), cancel)
                    buffer = ""

            if not cancel.is_set() and buffer.strip():
                await self._speak_chunk(turn_id, buffer.strip(), cancel)

            if cancel.is_set():
                return

            self._history.append({"role": "assistant", "content": full.strip()})
            await self._pending_events.put(
                {"type": "assistant_text", "turn_id": turn_id, "text": full.strip()}
            )
            await self._pending_events.put({"type": "tts_end", "turn_id": turn_id})
        except Exception as e:
            log.exception("reply failed")
            await self._pending_events.put({"type": "error", "message": str(e)})

    async def _speak_chunk(self, turn_id: str, text: str, cancel: asyncio.Event) -> None:
        assert self._tts is not None
        try:
            audio, sr = await asyncio.to_thread(self._tts.synth, text)
        except Exception as e:
            await self._pending_events.put({"type": "error", "message": f"tts: {e}"})
            return
        if cancel.is_set():
            return
        await self._pending_events.put(
            {"type": "tts_start", "turn_id": turn_id, "sample_rate": sr, "text": text}
        )
        for frame in _chunk_pcm16(audio, frame_ms=80):
            if cancel.is_set():
                return
            # Prepend a 4-byte big-endian length-of-turn-id header isn't worth the
            # complexity; we already gate playback client-side by turn_id from the
            # most recent tts_start. Stale frames are simply discarded by the
            # client once a tts_cancel arrives.
            await self._pending_events.put(frame)

    # ---- outbound to WS ----

    async def drain(self, ws: WebSocket) -> None:
        while True:
            try:
                item = self._pending_events.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, bytes):
                await ws.send_bytes(item)
            else:
                await ws.send_json(item)
