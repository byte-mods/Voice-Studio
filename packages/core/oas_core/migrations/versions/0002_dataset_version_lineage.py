"""Add ``parent_version_id`` to ``dataset_versions`` to record fork provenance.

A forked version points at the version it was derived from. ON DELETE SET NULL
because lineage is a provenance record, not a structural dependency — the fork
must outlive its ancestor.

The baseline migration (``0001_baseline``) uses ``Base.metadata.create_all``,
which means a fresh DB already has every column currently declared on the
ORM models — including ``parent_version_id``. This migration is therefore
idempotent by design: it only adds the column / index / FK when missing, so
it is a no-op on fresh databases and a real upgrade on databases stamped at
0001 before the column existed.

Revision ID: 0002_dataset_version_lineage
Revises: 0001_baseline
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_dataset_version_lineage"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def _has_column(table: str, column: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(c["name"] == column for c in inspector.get_columns(table))


def _has_index(table: str, index: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return any(ix["name"] == index for ix in inspector.get_indexes(table))


def upgrade() -> None:
    add_column = not _has_column("dataset_versions", "parent_version_id")
    add_index = not _has_index("dataset_versions", "ix_dataset_versions_parent_version_id")
    if not (add_column or add_index):
        return
    with op.batch_alter_table("dataset_versions") as batch:
        if add_column:
            batch.add_column(
                sa.Column("parent_version_id", sa.String(length=32), nullable=True)
            )
            batch.create_foreign_key(
                "fk_dataset_versions_parent_version_id",
                "dataset_versions",
                ["parent_version_id"],
                ["id"],
                ondelete="SET NULL",
            )
        if add_index:
            batch.create_index(
                "ix_dataset_versions_parent_version_id", ["parent_version_id"]
            )


def downgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch:
        batch.drop_constraint("fk_dataset_versions_parent_version_id", type_="foreignkey")
        if _has_index("dataset_versions", "ix_dataset_versions_parent_version_id"):
            batch.drop_index("ix_dataset_versions_parent_version_id")
        batch.drop_column("parent_version_id")
