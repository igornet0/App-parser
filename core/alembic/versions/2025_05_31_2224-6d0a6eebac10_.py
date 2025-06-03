"""empty message

Revision ID: 6d0a6eebac10
Revises: 304b77a79343
Create Date: 2025-05-31 22:24:35.468849

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6d0a6eebac10'
down_revision: Union[str, None] = '304b77a79343'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('users', 'user_telegram_id',
               existing_type=sa.BIGINT(),
               nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('users', 'user_telegram_id',
               existing_type=sa.BIGINT(),
               nullable=False)
