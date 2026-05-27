"""Native audio S2S session (Qwen2.5-Omni-style audio-LM).

Wire protocol is identical to `S2SSession` so the client and the WebSocket
endpoint don't need to know which mode is in play.

Differences from pipeline mode:
- Single multimodal model handles ASR + dialog + (optionally) TTS-style audio
  tokens. Examples: `Qwen/Qwen2.5-Omni-7B`, `meta-llama/LLaMA-Omni-1`,
  `THUDM/glm-4-voice-9b`.
- Audio input goes straight into the model's audio tower; text output is
  spoken via an attached TTS (codec vocoder or a fallback HF TTS).
- We still chunk audio by silence — *future* iterations will let the model
  perform its own VAD inside the audio tower.

For v1 this class wraps a chat-style audio-LM:
- accepts the user's audio as a single multimodal turn,
- generates a text reply (audio-token decoding is gated behind a model-
  specific flag),
- routes the text reply through a configurable fallback TTS for playback.

If the chosen base model emits audio tokens natively (Qwen2.5-Omni audio
output, GLM-4-Voice tokens), the `audio_decoder` path inside `_generate` is
the right place to plug in the per-model vocoder.
"""

from __future__ import annotations

import asyncio
import io
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket
from oas_core.db import ModelVersion, S2SPipeline, session_scope

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

CLIENT_SR = 16000


@dataclass
class NativeConfig:
    pipeline_id: str
    audio_lm_uri: str
    audio_lm_format: str
    tts_uri: str | None
    tts_format: str | None
    system_prompt: str
    vad: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)


def resolve_native(pipeline_id: str) -> NativeConfig:
    """Resolve pipeline rows for native-mode operation.

    We re-use `llm_*` slots on `S2SPipeline` for the audio-LM and `tts_*` slots
    as the fallback vocoder. The mode is selected in `runtime_config['mode']`.
    """
    with session_scope() as s:
        p = s.get(S2SPipeline, pipeline_id)
        if not p:
            raise ValueError(f"pipeline {pipeline_id!r} not found")
        runtime = dict(p.runtime_config or {})
        mode = runtime.get("mode", "pipeline")
        if mode != "native":
            raise ValueError(f"pipeline {pipeline_id!r} not in native mode (mode={mode!r})")

        if p.llm_version_id:
            v = s.get(ModelVersion, p.llm_version_id)
            if v is None:
                raise ValueError(f"audio-LM ModelVersion {p.llm_version_id!r} not found")
            audio_lm_uri, audio_lm_fmt = v.artifact_uri, v.format
        else:
            if not p.llm_fallback:
                raise ValueError("no audio-LM configured (llm_version_id + llm_fallback both empty)")
            audio_lm_uri, audio_lm_fmt = f"hf://{p.llm_fallback}", "hf"

        tts_uri: str | None = None
        tts_fmt: str | None = None
        if p.tts_version_id:
            v = s.get(ModelVersion, p.tts_version_id)
            if v is not None:
                tts_uri, tts_fmt = v.artifact_uri, v.format
        elif p.tts_fallback:
            tts_uri, tts_fmt = f"hf://{p.tts_fallback}", "hf"

        return NativeConfig(
            pipeline_id=p.id,
            audio_lm_uri=audio_lm_uri,
            audio_lm_format=audio_lm_fmt,
            tts_uri=tts_uri,
            tts_format=tts_fmt,
            system_prompt=p.system_prompt or "",
            vad=p.vad_config or {},
            runtime=runtime,
        )


# ---------------------------------------------------------------------------


class _AudioLM:
    """Multimodal audio-text LM.

    Loads via `AutoModelForCausalLM` / `AutoProcessor` and accepts a list of
    messages where audio turns carry numpy float32 arrays.

    For Qwen-Omni-style models the `.generate()` call can return both text and
    audio token sequences when `output_audio=True` (or whatever the model
    exposes). We attempt that path first via `reply_with_audio` and fall back
    to text-only `reply`.
    """

    def __init__(self, uri: str, fmt: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        target = uri.removeprefix("hf://").removeprefix("file://")
        self.processor = AutoProcessor.from_pretrained(target, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(target, torch_dtype=dtype, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        # Sniff capabilities once so the session can pick a path without
        # re-probing each turn.
        self.supports_audio_out = (
            hasattr(self.model, "generate")
            and "audio" in str(getattr(self.model.generate, "__doc__", "") or "").lower()
        ) or hasattr(self.model, "audio_decoder")

    def reply_with_audio(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        audio_in: Any,
        sample_rate: int,
        max_new_tokens: int = 200,
    ) -> tuple[str, np.ndarray | None, int | None]:
        """Generate a reply that includes synthesized audio when supported.

        Returns (text, audio, sample_rate). `audio is None` means the caller
        should fall back to TTS.
        """
        import torch

        messages = self._build_messages(system_prompt, history, audio_in, sample_rate)
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
            )
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = {"input_ids": inputs.to(self.device)}
        except Exception:
            inputs = self.processor(
                text=f"{system_prompt}\n[user audio]\n",
                audios=[audio_in],
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9
        )
        if self.supports_audio_out:
            # Models that accept these flags will produce audio; ones that don't
            # ignore them. We use a try/except to handle both shapes.
            gen_kwargs.update({"return_audio": True, "output_audio": True})

        with torch.no_grad():
            try:
                out = self.model.generate(**inputs, **gen_kwargs)
            except TypeError:
                # Older signatures reject the audio kwargs — retry without.
                gen_kwargs.pop("return_audio", None)
                gen_kwargs.pop("output_audio", None)
                out = self.model.generate(**inputs, **gen_kwargs)

        text = self._extract_text(out, inputs)
        audio, sr = self._extract_audio(out)
        return text, audio, sr

    def _build_messages(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        audio_in: Any,
        sample_rate: int,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        for turn in history:
            messages.append(
                {
                    "role": turn["role"],
                    "content": [{"type": "text", "text": turn["content"]}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_in, "sampling_rate": sample_rate}],
            }
        )
        return messages

    def _extract_text(self, out: Any, inputs: dict[str, Any]) -> str:
        # Output may be a Tensor (token ids), a tuple (ids, audio), or a dict-like.
        if isinstance(out, tuple):
            ids = out[0]
        elif hasattr(out, "sequences"):
            ids = out.sequences
        elif hasattr(out, "input_ids"):
            ids = out.input_ids
        else:
            ids = out
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen = ids[0, input_len:] if input_len else ids[0]
        return str(self.processor.batch_decode([gen], skip_special_tokens=True)[0].strip())

    def _extract_audio(self, out: Any) -> tuple[Any, int | None]:
        """Best-effort audio extraction across model families.

        Returns (np.ndarray | None, sample_rate | None). The caller falls back
        to TTS when audio is None.
        """
        import numpy as np

        audio = None
        sr = None
        # Qwen-Omni style: returns a dict-like with 'audios' and 'sampling_rate'.
        if hasattr(out, "audios") and out.audios is not None:
            audio = out.audios[0]
            sr = int(getattr(out, "sampling_rate", 24000))
        elif isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            audio = out[1]
            sr = 24000
        elif isinstance(out, dict) and "audio" in out:
            audio = out["audio"]
            sr = int(out.get("sampling_rate", 24000))

        if audio is None:
            return None, None
        # Coerce torch tensors / lists to numpy float32 mono.
        try:
            import torch

            if isinstance(audio, torch.Tensor):
                audio = audio.detach().to("cpu", dtype=torch.float32).numpy()
        except Exception:
            pass
        arr = np.asarray(audio, dtype=np.float32).squeeze()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr, sr

    def reply(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        audio: Any,
        sample_rate: int,
        max_new_tokens: int = 200,
    ) -> str:
        """Send a multimodal turn (audio + prior text history) and return text."""
        import torch

        # Build a messages structure the processor understands. Each model
        # family has its own schema; we attempt the Qwen-Omni style first and
        # fall back to a generic system+user prompt that embeds an `<audio>`
        # placeholder.
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        for turn in history:
            messages.append(
                {
                    "role": turn["role"],
                    "content": [{"type": "text", "text": turn["content"]}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio, "sampling_rate": sample_rate}],
            }
        )

        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
            )
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = {"input_ids": inputs.to(self.device)}
        except Exception:
            # Generic fallback: serialize audio to wav bytes and ask the
            # processor to tokenize a `<|audio|>`-prefixed text prompt.
            inputs = self.processor(
                text=f"{system_prompt}\n[user audio]\n",
                audios=[audio],
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        gen = out[0, input_len:] if input_len else out[0]
        return str(self.processor.batch_decode([gen], skip_special_tokens=True)[0].strip())


class _FallbackTTS:
    """Optional vocoder for native sessions whose audio-LM only emits text."""

    def __init__(self, uri: str, fmt: str) -> None:
        self._piper = None
        self._hf = None
        target_path = uri.removeprefix("file://")
        if fmt == "piper-onnx":
            from piper import PiperVoice

            voice_dir = Path(target_path)
            self._piper = PiperVoice.load(str(next(voice_dir.glob("*.onnx"))))
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


def _to_pcm16(arr: Any) -> bytes:
    import numpy as np

    clipped = np.clip(arr, -1.0, 1.0)
    return bytes((clipped * 32767.0).astype("<i2").tobytes())


def _from_pcm16(buf: bytes) -> Any:
    import numpy as np

    return np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0


def _chunk_pcm16(arr: Any, frame_ms: int = 80) -> list[bytes]:
    n = max(1, int(CLIENT_SR * frame_ms / 1000))
    out: list[bytes] = []
    for i in range(0, len(arr), n):
        out.append(_to_pcm16(arr[i : i + n]))
    return out


class NativeS2SSession:
    """WebSocket-side session compatible with `S2SSession`."""

    def __init__(self, cfg: NativeConfig) -> None:
        self.cfg = cfg
        vad = cfg.vad or {}
        self.silence_tail_ms = int(vad.get("silence_tail_ms", 600))
        self.min_speech_ms = int(vad.get("min_speech_ms", 200))
        self.rms_threshold = float(vad.get("rms_threshold", 0.015))

        self._audio_lm: _AudioLM | None = None
        self._tts: _FallbackTTS | None = None
        self._inbox: list[bytes] = []
        self._pending_events: asyncio.Queue[dict[str, Any] | bytes] = asyncio.Queue()
        self._history: list[dict[str, str]] = []  # text-only chat history
        self._silence_ms = 0
        self._speech_ms = 0
        self._turn_id: str | None = None
        self._reply_turn_id: str | None = None
        self._cancel_event: asyncio.Event | None = None
        self._reply_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._pending_events.put(
            {"type": "ready", "pipeline_id": self.cfg.pipeline_id, "mode": "native"}
        )

    async def close(self) -> None:
        await self._cancel_in_flight(send_event=False)

    async def _ensure_loaded(self) -> None:
        if self._audio_lm is not None and (self.cfg.tts_uri is None or self._tts is not None):
            return
        await self._pending_events.put({"type": "loading"})
        await asyncio.to_thread(self._load_blocking)
        await self._pending_events.put({"type": "loaded"})

    def _load_blocking(self) -> None:
        if self._audio_lm is None:
            log.info("loading audio-LM: %s", self.cfg.audio_lm_uri)
            self._audio_lm = _AudioLM(self.cfg.audio_lm_uri, self.cfg.audio_lm_format)
        if self._tts is None and self.cfg.tts_uri is not None:
            log.info("loading fallback TTS: %s", self.cfg.tts_uri)
            self._tts = _FallbackTTS(self.cfg.tts_uri, self.cfg.tts_format or "hf")

    async def push_audio(self, chunk: bytes) -> None:
        import numpy as np

        samples = _from_pcm16(chunk)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples**2)))
        chunk_ms = int(1000 * samples.size / CLIENT_SR)
        is_speech = rms >= self.rms_threshold

        if is_speech:
            if self._turn_id is None:
                self._turn_id = secrets.token_urlsafe(6)
                await self._pending_events.put({"type": "turn_start", "turn_id": self._turn_id})
                if self._reply_task is not None and not self._reply_task.done():
                    await self._cancel_in_flight()
            self._inbox.append(chunk)
            self._speech_ms += chunk_ms
            self._silence_ms = 0
        else:
            if self._turn_id is None:
                return
            self._inbox.append(chunk)
            self._silence_ms += chunk_ms
            if self._speech_ms >= self.min_speech_ms and self._silence_ms >= self.silence_tail_ms:
                await self._finalize_turn()

    async def on_control(self, raw: str) -> None:
        import json

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "end_turn":
            await self._finalize_turn()
        elif msg.get("type") == "reset":
            await self._cancel_in_flight()
            self._history = []
            await self._pending_events.put({"type": "reset"})

    async def _finalize_turn(self) -> None:
        if not self._inbox or self._turn_id is None:
            return
        turn_id = self._turn_id
        chunks = self._inbox
        self._inbox = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._turn_id = None

        try:
            await self._ensure_loaded()
        except Exception as e:
            await self._pending_events.put({"type": "error", "message": str(e)})
            return

        audio = _from_pcm16(b"".join(chunks))
        self._cancel_event = asyncio.Event()
        self._reply_turn_id = turn_id
        self._reply_task = asyncio.create_task(
            self._run_reply(turn_id, audio, self._cancel_event)
        )

    async def _cancel_in_flight(self, *, send_event: bool = True) -> None:
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
            await self._pending_events.put({"type": "tts_cancel", "turn_id": self._reply_turn_id})
        self._reply_turn_id = None

    async def _run_reply(self, turn_id: str, audio: Any, cancel: asyncio.Event) -> None:
        try:
            assert self._audio_lm is not None

            # Prefer the audio-out path; the wrapper falls back to text-only
            # if the model doesn't support audio output.
            reply_text, audio_out, sr = await asyncio.to_thread(
                self._audio_lm.reply_with_audio,
                self.cfg.system_prompt,
                self._history,
                audio,
                CLIENT_SR,
            )
            if cancel.is_set():
                return

            self._history.append({"role": "user", "content": "[audio]"})
            self._history.append({"role": "assistant", "content": reply_text})

            await self._pending_events.put(
                {"type": "final_transcript", "turn_id": turn_id, "text": "[audio]"}
            )
            await self._pending_events.put(
                {"type": "assistant_text", "turn_id": turn_id, "text": reply_text}
            )

            # Audio source priority: native model output > fallback TTS > silence.
            if audio_out is not None and sr is not None:
                await self._pending_events.put(
                    {
                        "type": "tts_start",
                        "turn_id": turn_id,
                        "sample_rate": sr,
                        "text": reply_text,
                        "source": "native",
                    }
                )
                for frame in _chunk_pcm16(audio_out, frame_ms=80):
                    if cancel.is_set():
                        return
                    await self._pending_events.put(frame)
            elif self._tts is not None and reply_text.strip():
                fallback_audio, fallback_sr = await asyncio.to_thread(self._tts.synth, reply_text)
                if cancel.is_set():
                    return
                await self._pending_events.put(
                    {
                        "type": "tts_start",
                        "turn_id": turn_id,
                        "sample_rate": fallback_sr,
                        "text": reply_text,
                        "source": "fallback",
                    }
                )
                for frame in _chunk_pcm16(fallback_audio, frame_ms=80):
                    if cancel.is_set():
                        return
                    await self._pending_events.put(frame)

            await self._pending_events.put({"type": "tts_end", "turn_id": turn_id})
        except Exception as e:
            log.exception("native reply failed")
            await self._pending_events.put({"type": "error", "message": str(e)})

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
