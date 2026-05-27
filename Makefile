.PHONY: help install install-py install-web dev dev-server dev-web test lint typecheck fmt build clean

help:
	@echo "Open Audio Studio dev targets:"
	@echo "  install        Install all Python + Node deps"
	@echo "  dev            Run server + web together"
	@echo "  dev-server     Run FastAPI backend on :8000"
	@echo "  dev-web        Run Next.js frontend on :3000"
	@echo "  test           Run all tests"
	@echo "  lint           Lint everything"
	@echo "  typecheck      mypy + tsc"
	@echo "  fmt            Format everything"
	@echo "  build          Build production artifacts"
	@echo "  clean          Clean build/cache"

install: install-py install-web

install-py:
	python -m pip install --upgrade pip
	pip install -e packages/core -e packages/sdk -e apps/server
	pip install pytest pytest-asyncio ruff mypy httpx

install-web:
	cd apps/web && pnpm install

dev:
	@echo "Starting server on :8000 and web on :3000"
	@(trap 'kill 0' SIGINT; \
	  $(MAKE) dev-server & \
	  $(MAKE) dev-web & \
	  wait)

dev-server:
	cd apps/server && uvicorn oas_server.main:app --reload --host 0.0.0.0 --port 8000

dev-web:
	cd apps/web && pnpm dev

test:
	pytest

lint:
	ruff check .
	cd apps/web && pnpm lint || true

typecheck:
	mypy packages apps || true
	cd apps/web && pnpm typecheck || true

fmt:
	ruff check --fix .
	ruff format .

build:
	cd apps/web && pnpm build

gen-api-client:
	./scripts/gen_openapi_client.sh

db-upgrade:
	cd packages/core && alembic upgrade head

db-revision:
	@test -n "$(m)" || (echo "Usage: make db-revision m='your message'" && exit 1)
	cd packages/core && alembic revision --autogenerate -m "$(m)"

db-downgrade:
	cd packages/core && alembic downgrade -1

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf apps/web/.next apps/web/node_modules dist build *.egg-info
