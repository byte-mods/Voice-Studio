"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from oas_core.db import init_db
from oas_core.observability import configure_logging
from oas_core.queue import WorkerPool
from oas_core.queue.backend import JobBackend, list_handlers
from oas_core.settings import Settings, get_settings

from oas_server import handlers as _handlers  # noqa: F401 — registers job handlers
from oas_server.middleware.audit import AuditMiddleware
from oas_server.middleware.rate_limit import RateLimitMiddleware
from oas_server.middleware.request_id import RequestIDMiddleware
from oas_server.routers import (
    audit as audit_router,
)
from oas_server.routers import (
    auth as auth_router,
)
from oas_server.routers import (
    datasets,
    experiments,
    health,
    jobs,
    kernels,
    models,
    plans,
    projects,
    s2s,
    serve,
    system,
    uploads,
    ws,
)
from oas_server.routers import (
    settings as settings_router,
)


def _configure_logging(level: str) -> None:
    configure_logging(level)


def _build_backend(settings: Settings) -> JobBackend:
    """Choose the job backend per OAS_JOB_BACKEND."""
    backend = settings.job_backend.lower()
    if backend == "ray":
        from oas_core.queue.ray_backend import RayBackend

        return RayBackend(address=settings.ray_address)
    if backend == "modal":
        from oas_core.queue.modal_backend import ModalBackend

        return ModalBackend()
    if backend == "slurm":
        from oas_core.queue.slurm_backend import SlurmBackend

        return SlurmBackend()
    return WorkerPool()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _configure_logging(settings.log_level)
    init_db(settings)
    # Discover plugins eagerly so the UI shows their handler names on startup.
    list_handlers()
    backend = _build_backend(settings)
    if hasattr(backend, "start"):
        backend.start()
    app.state.worker_pool = backend
    try:
        yield
    finally:
        if hasattr(backend, "stop"):
            backend.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Open Audio Studio API",
        version="0.1.0",
        description="REST + WebSocket API for the Open Audio Studio.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Starlette runs middlewares in reverse-of-add order. We want the request
    # id installed first (so audit + rate limit logs can use it), then audit,
    # then the rate limiter as the outermost gate.
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(health.router)
    app.include_router(system.router)
    app.include_router(settings_router.router)
    app.include_router(auth_router.router)
    app.include_router(audit_router.router)
    app.include_router(projects.router)
    app.include_router(datasets.router)
    app.include_router(jobs.router)
    app.include_router(experiments.router)
    app.include_router(models.router)
    app.include_router(kernels.router)
    app.include_router(s2s.router)
    app.include_router(serve.router)
    app.include_router(plans.router)
    app.include_router(uploads.router)
    app.include_router(ws.router)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "oas_server.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
    )
