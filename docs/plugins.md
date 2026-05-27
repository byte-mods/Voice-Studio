# Plugins

Open Audio Studio discovers job handlers via Python entry points. Any package
installed in the server's Python environment can contribute new job kinds
without changes to the core.

## Quickstart

`plugins/example/` ships a working reference plugin. To register it:

```bash
pip install -e plugins/example
# restart the server
```

The studio's `/jobs/handlers` endpoint will now include `echo_plus`, and any
`POST /jobs` with `{"kind": "echo_plus", ...}` will route to it.

## Writing a plugin

Create a normal Python package with this in your `pyproject.toml`:

```toml
[project.entry-points."oas.handlers"]
my_kind = "my_pkg.handlers:my_handler"
```

`my_handler` must be a callable `JobHandler` matching the protocol in
`oas_core.queue.backend`:

```python
from oas_core.queue.backend import JobContext

def my_handler(ctx: JobContext) -> dict:
    ctx.log("doing work")
    ctx.heartbeat()
    return {"some_metric": 1.23}
```

`ctx.config` is the job's submitted config dict. `ctx.log(...)` streams to
the run's log file, which the UI tails over WebSocket. Handler logs that
match the pattern `step k=v k=v` are auto-rendered as live charts on the job
detail page.

## Discovery rules

- Entry points in group `oas.handlers` are loaded on the first call to
  `list_handlers()` or `get_handler(...)`.
- Built-in handlers register before plugin discovery, so a plugin cannot
  override a built-in name. (Override needs to be explicit; raise an issue if
  you need this.)
- Discovery is best-effort: a single misbehaving plugin can't crash the
  server — its load exception is swallowed and logged.

## Job backends

The same plugin mechanism powers swappable job backends. Set
`OAS_JOB_BACKEND=ray` (or `inprocess`, the default) at server startup.
Built-ins:

| Backend     | Module                              | Notes |
|-------------|-------------------------------------|-------|
| `inprocess` | `oas_core.queue.worker.WorkerPool`  | Local thread pool. Default. |
| `ray`       | `oas_core.queue.ray_backend.RayBackend` | Connects to a Ray cluster (`OAS_RAY_ADDRESS=auto` or a head node URL). |
| `modal`     | `oas_core.queue.modal_backend.ModalBackend` | Runs each job in a Modal container (`modal token new`). Requires a network-reachable DB (Postgres / MySQL). |
| `slurm`     | `oas_core.queue.slurm_backend.SlurmBackend` | `sbatch`s each job. Controlled by `OAS_SLURM_PARTITION`, `OAS_SLURM_GPUS`, `OAS_SLURM_TIME`, `OAS_SLURM_CPUS`, `OAS_SLURM_MEM`, `OAS_SLURM_EXTRA`, `OAS_SLURM_PYTHON`, `OAS_SLURM_SETUP`. |

K8s + RunPod adapters follow the same pattern and can be added as plugins.
