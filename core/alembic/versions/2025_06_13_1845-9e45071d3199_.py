"""empty message

Revision ID: 9e45071d3199
Revises: 5568ab957685
Create Date: 2025-06-13 18:45:06.831815

"""
from typing import Sequence, Union
from sqlalchemy.dialects import postgresql
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9e45071d3199'
down_revision: Union[str, None] = '5568ab957685'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade():
    op.alter_column(
        'agent_feature_values',
        'value',
        type_=postgresql.JSONB,
        postgresql_using='to_jsonb(value::text)'
    )

def downgrade():
    op.alter_column(
        'agent_feature_values',
        'value',
        type_=sa.String,
        postgresql_using='value::text'
    )
