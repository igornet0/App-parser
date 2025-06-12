"""empty message

Revision ID: 55c5dd3f5a82
Revises: c62c15315e5f
Create Date: 2025-06-11 04:17:57.001723

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '55c5dd3f5a82'
down_revision: Union[str, None] = 'c62c15315e5f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.add_column('agents', sa.Column('name', sa.String(length=50), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column('agents', 'name')

