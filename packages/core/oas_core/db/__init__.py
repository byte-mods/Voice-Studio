"""Database layer: SQLAlchemy 2.x models + session factory.

Models mirror the studio's persistent entities: projects, datasets and their
versions, jobs and their runs, experiments, and registered models.
"""

from oas_core.db.engine import (
    SessionLocal,
    create_all,
    get_engine,
    get_session,
    init_db,
    session_scope,
)
from oas_core.db.models import (
    AuditLog,
    Base,
    Dataset,
    DatasetVersion,
    Experiment,
    Job,
    JobStatus,
    KernelDraft,
    Membership,
    Model,
    ModelSpec,
    ModelStage,
    ModelVersion,
    Project,
    Role,
    Run,
    RunStatus,
    S2SPipeline,
    User,
)

__all__ = [
    "AuditLog",
    "Base",
    "Dataset",
    "DatasetVersion",
    "Experiment",
    "Job",
    "JobStatus",
    "KernelDraft",
    "Membership",
    "Model",
    "ModelSpec",
    "ModelStage",
    "ModelVersion",
    "Project",
    "Role",
    "Run",
    "RunStatus",
    "S2SPipeline",
    "SessionLocal",
    "User",
    "create_all",
    "get_engine",
    "get_session",
    "init_db",
    "session_scope",
]
