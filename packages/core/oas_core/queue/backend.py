"""Job backend protocol + handler registry.

A `JobHandler` is a synchronous callable that receives a `JobContext` and
returns a metrics dict. Long-running handlers should periodically call
`ctx.heartbeat()` and `ctx.log()` to stream progress.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass
class JobContext:
    job_id: str
    run_id: str
    kind: str
    config: dict[str, Any]
    artifacts_dir: str
    logs_dir: str
    seed: int | None = None
    _cancelled: bool = False
    _log_sinks: list[Callable[[str], None]] = field(default_factory=list)

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def log(self, message: str) -> None:
        line = f"[{datetime.now(UTC).isoformat()}] {message}"
        for sink in self._log_sinks:
            with contextlib.suppress(Exception):
                sink(line)

    def add_log_sink(self, sink: Callable[[str], None]) -> None:
        self._log_sinks.append(sink)

    def heartbeat(self) -> None:
        # Hook for future progress reporting / liveness tracking.
        pass


class JobHandler(Protocol):
    def __call__(self, ctx: JobContext) -> dict[str, Any]: ...


class JobBackend(ABC):
    """Executes jobs. Implementations: in-process worker pool, Ray, Slurm, k8s."""

    @abstractmethod
    def submit(self, job_id: str) -> None:
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        ...


_HANDLERS: dict[str, JobHandler] = {}
_PLUGIN_DISCOVERY_DONE = False


def register_handler(kind: str, handler: JobHandler) -> None:
    if kind in _HANDLERS:
        raise ValueError(f"Handler for kind {kind!r} already registered")
    _HANDLERS[kind] = handler


def get_handler(kind: str) -> JobHandler:
    if kind not in _HANDLERS:
        _discover_plugins()
    if kind not in _HANDLERS:
        raise KeyError(f"No handler registered for kind {kind!r}")
    return _HANDLERS[kind]


def list_handlers() -> list[str]:
    _discover_plugins()
    return sorted(_HANDLERS.keys())


def _discover_plugins() -> None:
    """Discover handlers registered via the `oas.handlers` entry-point group.

    A plugin package declares::

        # pyproject.toml
        [project.entry-points."oas.handlers"]
        my_kind = "my_pkg.handlers:my_handler_callable"

    The callable is loaded lazily and registered under its entry-point name.
    Duplicate names from a plugin are ignored so the first wins (built-in
    handlers always register first via direct `register_handler` calls).
    """
    global _PLUGIN_DISCOVERY_DONE
    if _PLUGIN_DISCOVERY_DONE:
        return
    _PLUGIN_DISCOVERY_DONE = True
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="oas.handlers")
    except Exception:
        return
    for ep in eps:
        if ep.name in _HANDLERS:
            continue
        try:
            handler = ep.load()
        except Exception:
            continue
        if callable(handler):
            _HANDLERS[ep.name] = handler


def reset_plugin_discovery() -> None:
    """Reset the plugin discovery cache to force reload of newly installed packages."""
    global _PLUGIN_DISCOVERY_DONE
    _PLUGIN_DISCOVERY_DONE = False
