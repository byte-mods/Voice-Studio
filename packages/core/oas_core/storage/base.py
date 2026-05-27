"""Storage abstraction.

URI schemes:
- `file://<absolute-path>` for local filesystem.
- `s3://<bucket>/<key>` for S3-compatible object stores.
- `mem://<key>` for in-memory tests.

Implementations must be safe for concurrent reads and atomic for writes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import BinaryIO
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class StorageObject:
    uri: str
    size: int
    sha256: str | None = None
    content_type: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedURI:
    scheme: str
    container: str  # bucket / host / empty for local
    key: str       # path within the container

    @property
    def uri(self) -> str:
        if self.scheme == "file":
            return f"file://{self.key}"
        return f"{self.scheme}://{self.container}/{self.key}"


def parse_uri(uri: str) -> ParsedURI:
    parsed = urlparse(uri)
    scheme = parsed.scheme or "file"
    if scheme == "file":
        return ParsedURI(scheme="file", container="", key=parsed.path or uri.removeprefix("file://"))
    return ParsedURI(scheme=scheme, container=parsed.netloc, key=parsed.path.lstrip("/"))


class Storage(ABC):
    """Bytes-level storage interface. All methods take fully-qualified URIs."""

    @abstractmethod
    def put_bytes(self, uri: str, data: bytes, *, content_type: str | None = None) -> StorageObject:
        ...

    @abstractmethod
    def put_stream(
        self, uri: str, stream: BinaryIO, *, content_type: str | None = None
    ) -> StorageObject:
        ...

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes:
        ...

    @abstractmethod
    def open_stream(self, uri: str) -> BinaryIO:
        ...

    @abstractmethod
    def exists(self, uri: str) -> bool:
        ...

    @abstractmethod
    def stat(self, uri: str) -> StorageObject:
        ...

    @abstractmethod
    def delete(self, uri: str) -> None:
        ...

    @abstractmethod
    def list(self, prefix_uri: str) -> Iterator[StorageObject]:
        ...
