"""Audio / file upload endpoint.

Accepts multipart uploads, streams to LocalStorage, returns a `file://` URI
plus content hash. Audio probing (duration, sample rate) is best-effort and
only runs when soundfile is available."""

from __future__ import annotations

import io
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from oas_core.settings import get_settings
from oas_core.storage import LocalStorage

router = APIRouter(prefix="/uploads", tags=["uploads"])


def _probe_audio(data: bytes) -> dict[str, float | int] | None:
    try:
        import soundfile as sf  # lazy

        info = sf.info(io.BytesIO(data))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_s": float(info.frames) / float(info.samplerate or 1),
        }
    except Exception:
        return None


@router.post("")
async def upload(file: UploadFile = File(...), prefix: str = "uploads") -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(400, "filename required")

    settings = get_settings()
    storage = LocalStorage(settings.data_dir)

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")

    # Random subdir to avoid collisions / overwrites.
    bucket = secrets.token_urlsafe(6)
    rel = f"{prefix}/{bucket}/{file.filename}"
    uri = storage.uri_for(rel)
    obj = storage.put_bytes(uri, data, content_type=file.content_type)

    info: dict[str, Any] = {
        "uri": uri,
        "filename": file.filename,
        "size": obj.size,
        "sha256": obj.sha256,
        "content_type": file.content_type,
    }
    if file.content_type and file.content_type.startswith("audio/"):
        probe = _probe_audio(data)
        if probe:
            info["audio"] = probe
    return info


@router.get("/file")
def fetch_file(uri: str = Query(...)) -> FileResponse:
    """Stream a file:// URI back to the client.

    Restricted to paths inside the studio data directory to prevent arbitrary
    filesystem reads.
    """
    if not uri.startswith("file://"):
        raise HTTPException(400, "only file:// URIs supported")
    settings = get_settings()
    path = Path(uri.removeprefix("file://")).resolve()
    root = settings.data_dir.resolve()
    if root not in path.parents and path != root:
        raise HTTPException(403, "path outside data directory")
    if not path.exists() or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
