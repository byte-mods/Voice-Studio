# Open Audio Studio

A UI-driven, production-grade platform to fine-tune, full-tune, evaluate, and
ship state-of-the-art **ASR, LLM, TTS, and Speech-to-Speech** models — plus a
research lab to design new architectures on **JAX + PyTorch** with **custom
CUDA / Triton / Pallas kernels**.

See [ROADMAP.md](./ROADMAP.md) for the full product vision and phased plan.

## Monorepo Layout

```text
apps/
  web/        Next.js 14 frontend (App Router, Tailwind, shadcn/ui)
  server/     FastAPI backend (REST + WebSocket)
packages/
  core/       Shared Python core: storage, db, queue, registry, manifest
  sdk/        Python client SDK (mirrors every API call)
kernels/      Custom CUDA / Triton / Pallas kernels
plugins/      First-party and community plugins
infra/
  docker/     Dockerfiles (CUDA, CPU)
  compose/    docker-compose dev stacks
docs/         User and developer documentation
scripts/      Dev and ops scripts
tests/        Top-level integration tests (per-package tests live alongside)
```

## Quick Start (dev)

Requirements: Python 3.11+, Node 20+, pnpm 9+, (optional) NVIDIA GPU + CUDA 12.

```bash
# Install Python workspaces
pip install -e packages/core -e packages/sdk -e apps/server

# Install web app
cd apps/web && pnpm install && cd -

# Run the dev stack (server + web)
make dev
```

The web app runs on http://localhost:3000 and the API on http://localhost:8000.

## Status

Phase 0 — Foundation, plus partial Phases 1–5 scaffolding. See ROADMAP.md.

## Code Quality Baseline

The Python codebase enforces:

- **mypy strict = 0 errors** across `packages/` + `apps/` (75 source files).
- **ruff = 0** violations under the configured rule set
  (`E, F, I, B, UP, N, SIM, RUF`).
- **pytest** = 50 passing tests (server + core).

`.history/` and `.venv/` are excluded from both ruff and mypy via
`pyproject.toml`. The strict-mode baseline is load-bearing — any new code must
land annotated, and any new `# type: ignore` comment must carry a specific
error code (e.g. `# type: ignore[misc]`).

## API Surface

REST + WebSocket served from `apps/server/oas_server`. Routers:

- `health` — `/healthz`, `/readyz`
- `system` — runtime + handler discovery
- `auth` — signup, login, token issuance (PBKDF2 + HMAC-signed tokens)
- `projects`, `datasets`, `models`, `experiments`, `plans`, `jobs`
- `architectures`, `kernels` — Architecture Lab
- `s2s` — pipeline + native session orchestration
- `serve` — per-`ModelVersion` inference:
  `POST /serve/asr/{id}/transcribe`,
  `WS   /serve/asr/{id}/stream`,
  `POST /serve/llm/{id}/v1/chat/completions` (OpenAI-compatible, `stream=true` supported),
  `POST /serve/tts/{id}/synthesize`
- `uploads`, `settings`, `audit`, `ws`

Job backends: local `WorkerPool` (default), `RayBackend`, `ModalBackend`,
`SlurmBackend` — selected via `OAS_JOB_BACKEND`.
