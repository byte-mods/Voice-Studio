"""Storage backend factory.

Selects the right `Storage` implementation by URI scheme. Defaults to the local
filesystem rooted at `Settings.data_dir`.

Env knobs:
  OAS_STORAGE_URI=s3://my-bucket          -> use S3 with that bucket
  OAS_S3_REGION=us-east-1
  OAS_S3_ENDPOINT=https://...             -> for MinIO / R2
"""

from __future__ import annotations

import os
from functools import lru_cache

from oas_core.settings import get_settings
from oas_core.storage.base import Storage
from oas_core.storage.local import LocalStorage


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    target = os.environ.get("OAS_STORAGE_URI", "").strip()
    if target.startswith("s3://"):
        from oas_core.storage.s3 import S3Storage  # lazy import

        bucket = target.removeprefix("s3://").split("/", 1)[0]
        return S3Storage(
            bucket=bucket,
            region=os.environ.get("OAS_S3_REGION"),
            endpoint_url=os.environ.get("OAS_S3_ENDPOINT"),
        )
    settings = get_settings()
    return LocalStorage(settings.data_dir)


def reset_storage_cache() -> None:
    """Used by tests to force re-initialization."""
    get_storage.cache_clear()
