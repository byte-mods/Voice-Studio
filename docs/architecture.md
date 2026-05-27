# Architecture

High-level view of how Open Audio Studio is wired together.

## Components

```text
┌────────────────────────────────────────────────────────────┐
│ apps/web (Next.js 14)                                      │
│   - 6 modality sections + Jobs / Models / Experiments      │
│   - Calls /api/* which is rewritten to the FastAPI server  │
└──────────────────────────┬─────────────────────────────────┘
                           │ HTTP / WebSocket
┌──────────────────────────▼─────────────────────────────────┐
│ apps/server (FastAPI)                                      │
│   - REST routers: projects, datasets, jobs, models, ...    │
│   - WS: /ws/runs/{run_id}/logs                              │
│   - Starts an in-process WorkerPool on lifespan startup     │
└──────────────────────────┬─────────────────────────────────┘
                           │ Python
┌──────────────────────────▼─────────────────────────────────┐
│ packages/core (oas_core)                                   │
│   manifest/  - Dataset manifest v1 schema (ASR/TTS/LLM/S2S) │
│   storage/   - LocalStorage now; S3 adapter to follow      │
│   db/        - SQLAlchemy 2.x models + engine              │
│   queue/     - JobBackend protocol + in-process WorkerPool  │
│   registry/  - Model registry service                       │
│   settings/  - Typed settings from env                      │
└────────────────────────────────────────────────────────────┘

packages/sdk (oas) - Python SDK mirroring the REST API.
kernels/           - Custom CUDA / Triton / Pallas kernels (Phase 6).
plugins/           - Plugin SDK and first-party plugins (Phase 9).
infra/             - Dockerfiles and compose.
```

## Data flow: submitting a job

1. UI or SDK POSTs `/jobs` with `{ project_id, kind, name, config }`.
2. `routers/jobs.py` calls `oas_core.queue.submit_job`, which inserts a row
   into the `jobs` table with status `QUEUED` and returns the id.
3. The `WorkerPool`, started by the FastAPI lifespan, polls the DB for
   QUEUED jobs, atomically claims one (`SELECT … FOR UPDATE SKIP LOCKED`),
   creates a `Run` row, and looks up a handler from the registry by `kind`.
4. The handler runs on a worker thread, streaming logs via `JobContext.log`
   into `data/runs/{run_id}/logs/stdout.log`.
5. On completion, both `Run.status` and `Job.status` are updated, metrics are
   persisted, and artifacts land in `data/runs/{run_id}/artifacts/`.

## Adding a new job kind

```python
from oas_core.queue.backend import JobContext, register_handler

def my_handler(ctx: JobContext) -> dict:
    ctx.log("doing work")
    return {"metric": 1.23}

register_handler("my_kind", my_handler)
```

Place the registration in a module that `apps/server/oas_server/handlers.py`
imports at startup, and the new kind appears in `/jobs/handlers` and the
UI's job-submit form automatically.

## Storage URIs

The studio uses URIs everywhere artifacts are referenced. Supported schemes:

- `file://<absolute-path>` — local FS (default).
- `s3://<bucket>/<key>` — S3-compatible (planned).
- `mem://<key>` — in-memory for tests (planned).

The `Storage` abstraction provides atomic writes and streaming reads.

## End-to-end tests

`apps/web/e2e/` holds Playwright tests that drive the real UI against a
running backend.

```bash
# One-time
cd apps/web && pnpm install && pnpm e2e:install

# Backend on :8000, web on :3000 (started automatically by Playwright)
pnpm e2e
```

`e2e/happy_path.spec.ts` exercises create-project → create-dataset →
submit-noop-job → live log. `e2e/auth.spec.ts` covers the
`OAS_AUTH_REQUIRED=true` UX and is skipped by default; opt in by setting
`OAS_AUTH_REQUIRED_E2E=1`.

## Database migrations

Schema changes are managed with **Alembic**. The metadata target is
`oas_core.db.Base.metadata`; the SQLAlchemy URL is read from
`Settings.db_url` (same env var as the server).

```bash
# Upgrade to latest
make db-upgrade

# Create a new revision from current model state
make db-revision m="add foo column to bar"

# Roll back the last migration
make db-downgrade
```

The baseline migration `0001_baseline` is intentionally a single
`create_all` so existing single-user installs can adopt Alembic without
re-creating their data.

## Manifest layout

```text
<dataset-version-root>/
  manifest.json      # ManifestHeader
  samples.jsonl      # one Sample per line
  samples.parquet    # optional columnar mirror
```

See `packages/core/oas_core/manifest/schema.py` for the full schema.
