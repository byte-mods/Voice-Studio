"""SQLAlchemy engine + session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from oas_core.db.models import Base
from oas_core.settings import Settings, ensure_dirs, get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    ensure_dirs(settings)
    connect_args = {"check_same_thread": False} if settings.db_url.startswith("sqlite") else {}
    return create_engine(
        settings.db_url,
        future=True,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


SessionLocal = sessionmaker(bind=None, autoflush=False, autocommit=False, expire_on_commit=False)


def _ensure_factory_bound() -> None:
    if SessionLocal.kw.get("bind") is None:
        SessionLocal.configure(bind=get_engine())


def create_all(engine: Engine | None = None) -> None:
    Base.metadata.create_all(bind=engine or get_engine())


def init_db(settings: Settings | None = None) -> Engine:
    s = settings or get_settings()
    ensure_dirs(s)
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    return engine


def get_session() -> Session:
    _ensure_factory_bound()
    return SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    _ensure_factory_bound()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
