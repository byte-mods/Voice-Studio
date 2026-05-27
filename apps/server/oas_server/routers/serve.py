"""Inference serving endpoints per ModelVersion.

ASR:
  POST /serve/asr/{version_id}/transcribe   (multipart audio file)
  WS   /serve/asr/{version_id}/stream       (PCM16 16k mono frames → partials)

LLM:
  POST /serve/llm/{version_id}/v1/chat/completions  (OpenAI-compatible, supports stream=true)

TTS:
  POST /serve/tts/{version_id}/synthesize   ({"text": "..."}) → audio/wav
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from typing import Any, cast

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from oas_server.serving.cache import get_or_load
from oas_server.serving.loaders import load_asr, load_llm, load_tts

router = APIRouter(prefix="/serve", tags=["serve"])


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


@router.post("/asr/{version_id}/transcribe")
async def asr_transcribe(
    version_id: str,
    file: UploadFile = File(...),
    language: str | None = Query(default=None),
) -> dict[str, Any]:
    import io as _io

    import soundfile as sf

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    audio, sr = sf.read(_io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    try:
        srv = get_or_load(f"asr:{version_id}", lambda: load_asr(version_id))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return cast(dict[str, Any], srv.transcribe(audio, sr, language=language))


@router.websocket("/asr/{version_id}/stream")
async def asr_stream(ws: WebSocket, version_id: str) -> None:
    """Receive PCM16 16k mono frames; emit final transcript on disconnect.

    For v1 we transcribe in one shot when the client closes (or sends
    `{"type":"end"}`). Truly streaming partials require a token-level
    streaming ASR backend, which is the next iteration here.
    """
    import numpy as np

    await ws.accept()
    try:
        srv = get_or_load(f"asr:{version_id}", lambda: load_asr(version_id))
    except KeyError as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                buf.extend(msg["bytes"])
            elif msg.get("text"):
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "end":
                    break
    except WebSocketDisconnect:
        pass

    if not buf:
        await ws.send_json({"type": "final", "text": ""})
        await ws.close()
        return

    audio = np.frombuffer(bytes(buf), dtype="<i2").astype("float32") / 32768.0
    result = srv.transcribe(audio, 16000)
    await ws.send_json({"type": "final", **result})
    await ws.close()


# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionIn(BaseModel):
    model: str | None = None  # ignored; we already routed on version_id
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    base_model: str | None = Field(default=None, description="Required for PEFT adapter versions.")


def _openai_chunk(model_id: str, delta_text: str, finish: str | None = None) -> str:
    payload = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_text} if delta_text else {},
                "finish_reason": finish,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/llm/{version_id}/v1/chat/completions")
def llm_chat(version_id: str, body: ChatCompletionIn) -> Any:
    try:
        srv = get_or_load(
            f"llm:{version_id}:{body.base_model or ''}",
            lambda: load_llm(version_id, base_model=body.base_model),
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e)) from e

    messages = [m.model_dump() for m in body.messages]

    if body.stream:

        def gen() -> Iterator[str]:
            for piece in srv.stream(
                messages,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
            ):
                yield _openai_chunk(version_id, piece)
            yield _openai_chunk(version_id, "", finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    # Non-streaming: collect.
    text = "".join(
        list(
            srv.stream(
                messages,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
            )
        )
    )
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": version_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class SynthIn(BaseModel):
    text: str


@router.post("/tts/{version_id}/synthesize")
def tts_synth(version_id: str, body: SynthIn) -> Response:
    import soundfile as sf

    try:
        srv = get_or_load(f"tts:{version_id}", lambda: load_tts(version_id))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    audio, sr = srv.synth(body.text)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@router.post("/tts/{version_id}/stream")
def tts_stream(version_id: str, body: SynthIn) -> StreamingResponse:
    import base64
    import re
    import soundfile as sf

    try:
        srv = get_or_load(f"tts:{version_id}", lambda: load_tts(version_id))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e

    # Split into sentences to stream piece-by-piece
    parts = re.split(r'(?<=[.!?])\s+', body.text)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences and body.text.strip():
        sentences = [body.text.strip()]

    def gen() -> Iterator[str]:
        for sent in sentences:
            try:
                audio, sr = srv.synth(sent)
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
                wav_bytes = buf.getvalue()
                b64 = base64.b64encode(wav_bytes).decode("utf-8")
                payload = {"audio": b64, "text": sent}
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                err_payload = {"error": str(e), "text": sent}
                yield f"data: {json.dumps(err_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

