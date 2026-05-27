"""WebSocket endpoints for live job logs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from oas_core.db import Run, session_scope

router = APIRouter(prefix="/ws", tags=["ws"])


@router.websocket("/runs/{run_id}/logs")
async def stream_run_logs(ws: WebSocket, run_id: str) -> None:
    await ws.accept()
    with session_scope() as s:
        run = s.get(Run, run_id)
        if not run or not run.logs_uri:
            await ws.send_json({"error": "logs not found"})
            await ws.close()
            return
        path = Path(run.logs_uri.removeprefix("file://"))

    try:
        offset = 0
        while True:
            if path.exists():
                with path.open("rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                    if chunk:
                        await ws.send_text(chunk.decode("utf-8", errors="replace"))
                        offset += len(chunk)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
