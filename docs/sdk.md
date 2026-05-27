# Python SDK

The SDK mirrors every UI action so workflows are scriptable and reproducible.

## Install (from the monorepo)

```bash
pip install -e packages/sdk
```

## Hello world

```python
from oas import Studio

with Studio("http://localhost:8000") as s:
    project = s.projects.create(slug="demo", name="Demo")

    dataset = s.datasets.create(
        project.id, slug="asr1", name="ASR 1", modality="asr"
    )
    s.datasets.add_version(
        dataset.id, version="0.1.0",
        manifest_uri="file:///path/to/manifest",
        num_samples=100,
    )

    job = s.jobs.submit(project.id, kind="noop", name="warmup")
    final = s.jobs.wait(job.id, timeout=60)
    print(final.status, final.runs[0].metrics)

    model = s.models.create(project.id, "whisper-en", "Whisper EN", "asr")
    s.models.publish(
        model.id, "0.1.0",
        artifact_uri="file:///path/to/ckpt.safetensors",
        metrics={"wer": 0.12},
    )
    s.models.set_stage(version.id, "staging")
```

## Web TypeScript client

The web app currently uses hand-typed wrappers in `apps/web/src/lib/api.ts`.
A generated client is also available — run with the dev server up:

```bash
make gen-api-client   # writes apps/web/src/lib/api.generated.ts
```

The generated file is committed-friendly: `openapi-typescript` produces a
single deterministic `paths`/`components` type that can be imported alongside
the hand-typed module while it's incrementally adopted.

## Resource API

| Resource     | Methods                                                          |
| ------------ | ---------------------------------------------------------------- |
| `projects`   | `list`, `create`, `get`, `delete`                                |
| `datasets`   | `list`, `create`, `get`, `versions`, `add_version`               |
| `jobs`       | `kinds`, `list`, `submit`, `get`, `cancel`, `wait`               |
| `experiments`| `list`, `create`                                                 |
| `models`     | `list`, `create`, `versions`, `publish`, `set_stage`             |

## Testing your SDK code

Use FastAPI's `TestClient` to point the SDK at an in-process server:

```python
from fastapi.testclient import TestClient
from oas import Studio
from oas_server.main import create_app

client = TestClient(create_app())
s = Studio(client=client)
```
