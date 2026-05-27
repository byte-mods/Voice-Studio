from __future__ import annotations

from oas_core.queue.backend import (
    JobContext,
    list_handlers,
    register_handler,
)


def test_register_and_list() -> None:
    def my(ctx: JobContext) -> dict:
        return {}

    register_handler("plugin_test_kind", my)
    assert "plugin_test_kind" in list_handlers()


def test_duplicate_registration_rejected() -> None:
    def my(ctx: JobContext) -> dict:
        return {}

    register_handler("plugin_dup_kind", my)
    try:
        register_handler("plugin_dup_kind", my)
    except ValueError:
        return
    raise AssertionError("expected duplicate registration to raise ValueError")
