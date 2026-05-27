from __future__ import annotations

import io

from oas_core.storage import LocalStorage, get_storage, reset_storage_cache


def test_factory_defaults_to_local() -> None:
    reset_storage_cache()
    s = get_storage()
    assert isinstance(s, LocalStorage)


def test_factory_picks_s3_by_uri(monkeypatch) -> None:
    monkeypatch.setenv("OAS_STORAGE_URI", "s3://my-bucket")

    class _Stub:
        def __init__(self, *a, **kw):
            self.put_calls: list[dict] = []

        def put_object(self, **kw):
            self.put_calls.append(kw)

        def get_object(self, **kw):
            return {"Body": io.BytesIO(b"hello")}

        def head_object(self, **kw):
            return {"ContentLength": 5, "ContentType": "text/plain"}

        def delete_object(self, **kw):
            pass

        def get_paginator(self, name):
            return self

        def paginate(self, **kw):
            yield {"Contents": []}

    # Stub boto3 client by injecting via S3Storage constructor.
    from oas_core.storage.s3 import S3Storage

    s = S3Storage(bucket="my-bucket", client=_Stub())
    uri = s.uri_for("x.bin")
    obj = s.put_bytes(uri, b"hello", content_type="text/plain")
    assert obj.size == 5
    assert s.get_bytes(uri) == b"hello"
    info = s.stat(uri)
    assert info.size == 5


def test_factory_rejects_uri_with_wrong_bucket(monkeypatch) -> None:
    class _Stub:
        def put_object(self, **kw):
            pass

    from oas_core.storage.s3 import S3Storage

    s = S3Storage(bucket="b1", client=_Stub())
    try:
        s.put_bytes("s3://other-bucket/x", b"")
    except ValueError as e:
        assert "bucket" in str(e)
    else:
        raise AssertionError("expected ValueError")
