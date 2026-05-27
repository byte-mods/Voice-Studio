"""Pluggable artifact storage.

A `Storage` backend handles bytes and streams identified by URIs. The studio
uses storage for raw uploads, processed audio, manifests, checkpoints, and
arbitrary run artifacts. Local FS is the reference implementation; S3 ships as
an optional adapter behind `pip install 'oas-core[s3]'`.
"""

from oas_core.storage.base import Storage, StorageObject, parse_uri
from oas_core.storage.factory import get_storage, reset_storage_cache
from oas_core.storage.local import LocalStorage

__all__ = [
    "LocalStorage",
    "Storage",
    "StorageObject",
    "get_storage",
    "parse_uri",
    "reset_storage_cache",
]
