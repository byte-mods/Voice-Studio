"""Open Audio Studio Python SDK.

Mirrors the HTTP API so every UI action is scriptable::

    from oas import Studio

    studio = Studio("http://localhost:8000")
    project = studio.projects.create(slug="demo", name="Demo")
    dataset = studio.datasets.create(project.id, slug="asr1", name="ASR 1", modality="asr")
    job = studio.jobs.submit(project.id, kind="noop", name="warmup")
    studio.jobs.wait(job.id, timeout=30)
"""

from oas.client import Studio
from oas.types import (
    Dataset,
    DatasetVersion,
    Experiment,
    Job,
    Model,
    ModelVersion,
    Project,
    Run,
)

__all__ = [
    "Dataset",
    "DatasetVersion",
    "Experiment",
    "Job",
    "Model",
    "ModelVersion",
    "Project",
    "Run",
    "Studio",
]
__version__ = "0.1.0"
