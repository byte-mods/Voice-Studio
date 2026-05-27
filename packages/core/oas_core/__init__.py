"""Open Audio Studio — core library.

Public surface:

- `manifest`: dataset sample schemas and manifest readers/writers.
- `storage`: pluggable artifact storage (local FS, S3).
- `db`: SQLAlchemy models for projects, datasets, jobs, runs, models.
- `queue`: in-process job queue with worker pool.
- `registry`: model registry with versions, tags, and lineage.
- `settings`: typed settings loaded from env + config file.
"""

from oas_core import db, manifest, queue, registry, settings, storage

__all__ = ["db", "manifest", "queue", "registry", "settings", "storage"]
__version__ = "0.1.0"
