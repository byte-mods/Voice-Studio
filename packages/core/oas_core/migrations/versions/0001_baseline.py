"""Baseline schema — captures every table currently declared in
``oas_core.db.models.Base.metadata`` by simply calling ``create_all`` on the
connection's bind.

After this baseline, new migrations should be generated with::

    alembic -c packages/core/alembic.ini revision --autogenerate -m "..."

Revision ID: 0001_baseline
Revises:
"""

from __future__ import annotations

from alembic import op

from oas_core.db import Base

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
