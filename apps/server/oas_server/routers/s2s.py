"""Speech-to-speech pipeline CRUD + realtime WebSocket session."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from oas_core.db import Project, Role, S2SPipeline, session_scope
from pydantic import BaseModel, Field
from sqlalchemy import select

from oas_server.auth import CurrentUser, assert_role, require_user
from oas_server.s2s.native_session import NativeS2SSession, resolve_native
from oas_server.s2s.session import S2SSession, resolve_pipeline

log = logging.getLogger(__name__)

router = APIRouter(prefix="/s2s", tags=["s2s"])


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class PipelineIn(BaseModel):
    project_id: str
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-_]*$")
    name: str
    description: str | None = None
    asr_version_id: str | None = None
    llm_version_id: str | None = None
    tts_version_id: str | None = None
    asr_fallback: str | None = "openai/whisper-tiny"
    llm_fallback: str | None = "Qwen/Qwen2.5-0.5B-Instruct"
    tts_fallback: str | None = "tts_models/en/ljspeech/glow-tts"
    system_prompt: str | None = "You are a helpful spoken assistant. Keep replies brief."
    vad_config: dict[str, Any] = Field(default_factory=dict)
    runtime_config: dict[str, Any] = Field(default_factory=dict)


class PipelineOut(BaseModel):
    id: str
    project_id: str
    slug: str
    name: str
    description: str | None
    asr_version_id: str | None
    llm_version_id: str | None
    tts_version_id: str | None
    asr_fallback: str | None
    llm_fallback: str | None
    tts_fallback: str | None
    system_prompt: str | None
    vad_config: dict[str, Any]
    runtime_config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


def _to_out(p: S2SPipeline) -> PipelineOut:
    return PipelineOut(
        id=p.id,
        project_id=p.project_id,
        slug=p.slug,
        name=p.name,
        description=p.description,
        asr_version_id=p.asr_version_id,
        llm_version_id=p.llm_version_id,
        tts_version_id=p.tts_version_id,
        asr_fallback=p.asr_fallback,
        llm_fallback=p.llm_fallback,
        tts_fallback=p.tts_fallback,
        system_prompt=p.system_prompt,
        vad_config=p.vad_config or {},
        runtime_config=p.runtime_config or {},
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


@router.get("/pipelines", response_model=list[PipelineOut])
def list_pipelines(project_id: str | None = None) -> list[PipelineOut]:
    with session_scope() as s:
        stmt = select(S2SPipeline)
        if project_id:
            stmt = stmt.where(S2SPipeline.project_id == project_id)
        stmt = stmt.order_by(S2SPipeline.created_at.desc())
        return [_to_out(p) for p in s.scalars(stmt)]


@router.post("/pipelines", response_model=PipelineOut, status_code=status.HTTP_201_CREATED)
def create_pipeline(body: PipelineIn, user: CurrentUser = Depends(require_user)) -> PipelineOut:
    assert_role(user, body.project_id, Role.EDITOR)
    with session_scope() as s:
        if not s.get(Project, body.project_id):
            raise HTTPException(404, "project not found")
        existing = s.scalar(
            select(S2SPipeline).where(
                S2SPipeline.project_id == body.project_id, S2SPipeline.slug == body.slug
            )
        )
        if existing:
            raise HTTPException(409, "slug already in use")
        p = S2SPipeline(**body.model_dump())
        s.add(p)
        s.flush()
        return _to_out(p)


@router.get("/pipelines/{pipeline_id}", response_model=PipelineOut)
def get_pipeline(pipeline_id: str) -> PipelineOut:
    with session_scope() as s:
        p = s.get(S2SPipeline, pipeline_id)
        if not p:
            raise HTTPException(404)
        return _to_out(p)


@router.delete("/pipelines/{pipeline_id}", status_code=204)
def delete_pipeline(pipeline_id: str, user: CurrentUser = Depends(require_user)) -> None:
    with session_scope() as s:
        p = s.get(S2SPipeline, pipeline_id)
        if not p:
            raise HTTPException(404)
        assert_role(user, p.project_id, Role.ADMIN)
        s.delete(p)


# ---------------------------------------------------------------------------
# Realtime WebSocket
# ---------------------------------------------------------------------------


@router.websocket("/sessions/{pipeline_id}")
async def s2s_session(ws: WebSocket, pipeline_id: str) -> None:
    """Realtime speech-to-speech session.

    Wire protocol (client → server):
      - binary frames: raw PCM16, 16 kHz, mono, little-endian audio chunks.
      - text frame `{"type": "end_turn"}` to force endpointing.
      - text frame `{"type": "reset"}` to clear conversation history.

    Wire protocol (server → client):
      - text frame `{"type": "partial_transcript", "text": "..."}`
      - text frame `{"type": "final_transcript", "text": "..."}`
      - text frame `{"type": "assistant_text", "text": "..."}`
      - text frame `{"type": "tts_start", "sample_rate": 22050}`
      - binary frames: PCM16 audio chunks of the assistant's voice.
      - text frame `{"type": "tts_end"}`
      - text frame `{"type": "error", "message": "..."}`
    """
    await ws.accept()

    mode = "pipeline"
    try:
        with session_scope() as s:
            p = s.get(S2SPipeline, pipeline_id)
            if not p:
                raise ValueError(f"pipeline {pipeline_id!r} not found")
            mode = (p.runtime_config or {}).get("mode", "pipeline")
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return

    session: S2SSession | NativeS2SSession
    try:
        if mode == "native":
            session = NativeS2SSession(await asyncio.to_thread(resolve_native, pipeline_id))
        else:
            session = S2SSession(await asyncio.to_thread(resolve_pipeline, pipeline_id))
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close()
        return
    try:
        await session.start()
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                await session.push_audio(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                await session.on_control(msg["text"])
            # Drain any pending events from the session pipeline.
            await session.drain(ws)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("s2s session crashed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": "internal session error"})
    finally:
        await session.close()
