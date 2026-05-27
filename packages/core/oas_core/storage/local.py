"""Local filesystem storage backend.

Writes are atomic via temp-file + rename. Reads stream through `open(..., 'rb')`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from oas_core.storage.base import Storage, StorageObject, parse_uri


class LocalStorage(Storage):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- URI <-> path helpers ----

    def _path(self, uri: str) -> Path:
        parsed = parse_uri(uri)
        if parsed.scheme != "file":
            raise ValueError(f"LocalStorage cannot handle scheme {parsed.scheme!r}")
        p = Path(parsed.key)
        if not p.is_absolute():
            p = self.root / p
        # Prevent escaping the root with ../ traversal.
        resolved = p.resolve()
        if self.root not in resolved.parents and resolved != self.root:  # noqa: SIM102
            # Allow absolute paths the caller explicitly gave, but reject relative escapes.
            if not Path(parsed.key).is_absolute():
                raise ValueError(f"Refusing path outside storage root: {resolved}")
        return resolved

    def uri_for(self, relative: str | Path) -> str:
        return f"file://{(self.root / Path(relative)).resolve()}"

    # ---- Storage interface ----

    def put_bytes(
        self, uri: str, data: bytes, *, content_type: str | None = None
    ) -> StorageObject:
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        digest = hashlib.sha256(data).hexdigest()
        return StorageObject(uri=uri, size=len(data), sha256=digest, content_type=content_type)

    def put_stream(
        self, uri: str, stream: BinaryIO, *, content_type: str | None = None
    ) -> StorageObject:
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        size = 0
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise
        return StorageObject(uri=uri, size=size, sha256=h.hexdigest(), content_type=content_type)

    def get_bytes(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def open_stream(self, uri: str) -> BinaryIO:
        return self._path(uri).open("rb")

    def exists(self, uri: str) -> bool:
        try:
            return self._path(uri).exists()
        except ValueError:
            return False

    def stat(self, uri: str) -> StorageObject:
        path = self._path(uri)
        st = path.stat()
        return StorageObject(uri=uri, size=st.st_size)

    def delete(self, uri: str) -> None:
        path = self._path(uri)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    def list(self, prefix_uri: str) -> Iterator[StorageObject]:
        base = self._path(prefix_uri)
        if not base.exists():
            return
        if base.is_file():
            st = base.stat()
            yield StorageObject(uri=f"file://{base}", size=st.st_size)
            return
        for p in base.rglob("*"):
            if p.is_file():
                yield StorageObject(uri=f"file://{p}", size=p.stat().st_size)
