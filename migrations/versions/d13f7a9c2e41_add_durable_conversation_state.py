"""add durable conversation state

Revision ID: d13f7a9c2e41
Revises: af588838f85e
Create Date: 2026-08-04 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d13f7a9c2e41"
down_revision: Union[str, Sequence[str], None] = "af588838f85e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the JSON checkpoint fields to existing AI conversations."""

    with op.batch_alter_table("ai_conversations") as batch_op:
        batch_op.add_column(sa.Column("state_data", sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "checkpoint_version",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )
        batch_op.add_column(
            sa.Column(
                "state_updated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )


def downgrade() -> None:
    """Remove durable checkpoint fields."""

    with op.batch_alter_table("ai_conversations") as batch_op:
        batch_op.drop_column("state_updated_at")
        batch_op.drop_column("checkpoint_version")
        batch_op.drop_column("state_data")
