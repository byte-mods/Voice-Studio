"""S3-compatible storage backend.

Works against AWS S3, MinIO, Cloudflare R2, and anything else that speaks the
S3 API. Install with `pip install 'oas-core[s3]'`.

URIs are `s3://<bucket>/<key>`. The bucket is fixed per-instance so callers
can't accidentally write across tenants by crafting a URI.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from typing import Any, BinaryIO

from oas_core.storage.base import Storage, StorageObject, parse_uri


class S3Storage(Storage):
    def __init__(
        self,
        bucket: str,
        *,
        region: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        if client is not None:
            self._client = client
        else:
            import boto3  # type: ignore

            kwargs: dict[str, Any] = {}
            if region:
                kwargs["region_name"] = region
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            self._client = boto3.client("s3", **kwargs)

    def uri_for(self, key: str) -> str:
        return f"s3://{self.bucket}/{key.lstrip('/')}"

    def _key(self, uri: str) -> str:
        parsed = parse_uri(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"S3Storage can't handle scheme {parsed.scheme!r}")
        if parsed.container != self.bucket:
            raise ValueError(f"URI bucket {parsed.container!r} != configured {self.bucket!r}")
        return parsed.key

    # ---- Storage interface ----

    def put_bytes(
        self, uri: str, data: bytes, *, content_type: str | None = None
    ) -> StorageObject:
        key = self._key(uri)
        extra: dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)
        return StorageObject(
            uri=uri,
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            content_type=content_type,
        )

    def put_stream(
        self, uri: str, stream: BinaryIO, *, content_type: str | None = None
    ) -> StorageObject:
        key = self._key(uri)
        h = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            h.update(chunk)
            size += len(chunk)
        body = b"".join(chunks)
        extra: dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        self._client.put_object(Bucket=self.bucket, Key=key, Body=body, **extra)
        return StorageObject(uri=uri, size=size, sha256=h.hexdigest(), content_type=content_type)

    def get_bytes(self, uri: str) -> bytes:
        key = self._key(uri)
        obj = self._client.get_object(Bucket=self.bucket, Key=key)
        return bytes(obj["Body"].read())

    def open_stream(self, uri: str) -> BinaryIO:
        return io.BytesIO(self.get_bytes(uri))

    def exists(self, uri: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(uri))
            return True
        except Exception:
            return False

    def stat(self, uri: str) -> StorageObject:
        key = self._key(uri)
        head = self._client.head_object(Bucket=self.bucket, Key=key)
        return StorageObject(
            uri=uri,
            size=int(head.get("ContentLength", 0)),
            content_type=head.get("ContentType"),
        )

    def delete(self, uri: str) -> None:
        key = self._key(uri)
        # Single delete is fine; for prefixes the caller should iterate via list().
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def list(self, prefix_uri: str) -> Iterator[StorageObject]:
        prefix = self._key(prefix_uri)
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for entry in page.get("Contents", []):
                yield StorageObject(
                    uri=self.uri_for(entry["Key"]),
                    size=int(entry.get("Size", 0)),
                )
