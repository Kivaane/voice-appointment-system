"""add AI request idempotency

Revision ID: e72a4b8d1f30
Revises: d13f7a9c2e41
Create Date: 2026-08-04 00:00:01.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e72a4b8d1f30"
down_revision: Union[str, Sequence[str], None] = "d13f7a9c2e41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create durable request execution records."""

    op.create_table(
        "ai_request_executions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["ai_conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "request_id",
            name="uq_ai_request_execution_thread_request",
        ),
    )
    op.create_index(
        "ix_ai_request_executions_conversation_id",
        "ai_request_executions",
        ["conversation_id"],
    )
    op.create_index(
        "ix_ai_request_executions_request_id",
        "ai_request_executions",
        ["request_id"],
    )
    op.create_index(
        "ix_ai_request_executions_status",
        "ai_request_executions",
        ["status"],
    )


def downgrade() -> None:
    """Drop request idempotency records."""

    op.drop_index(
        "ix_ai_request_executions_status",
        table_name="ai_request_executions",
    )
    op.drop_index(
        "ix_ai_request_executions_request_id",
        table_name="ai_request_executions",
    )
    op.drop_index(
        "ix_ai_request_executions_conversation_id",
        table_name="ai_request_executions",
    )
    op.drop_table("ai_request_executions")
