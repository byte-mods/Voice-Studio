"""Per-version model cache.

Models are expensive to load and cheap to call. We keep an LRU of N loaded
models in a process-global dict, behind a lock so the FastAPI worker pool
can hit it concurrently. Eviction calls `.close()` if defined.
"""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

_LOCK = threading.RLock()
_CACHE: OrderedDict[str, Any] = OrderedDict()
_MAX = 4


def get_or_load(key: str, loader: Callable[[], Any]) -> Any:
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    obj = loader()  # loader may take seconds; don't hold the lock
    with _LOCK:
        _CACHE[key] = obj
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX:
            _, evicted = _CACHE.popitem(last=False)
            close = getattr(evicted, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        return obj


def set_max(n: int) -> None:
    global _MAX
    with _LOCK:
        _MAX = max(1, int(n))


def clear() -> None:
    with _LOCK:
        for obj in _CACHE.values():
            close = getattr(obj, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        _CACHE.clear()
