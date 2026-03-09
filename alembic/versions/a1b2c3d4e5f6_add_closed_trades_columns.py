"""Add regime, features, kelly_fraction to closed_trades

Revision ID: a1b2c3d4e5f6
Revises: 2b96bec0820a
Create Date: 2026-03-09 01:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '2b96bec0820a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite doesn't support IF NOT EXISTS for ADD COLUMN, so use batch mode
    with op.batch_alter_table('closed_trades') as batch_op:
        batch_op.add_column(sa.Column('regime', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('features', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('kelly_fraction', sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('closed_trades') as batch_op:
        batch_op.drop_column('kelly_fraction')
        batch_op.drop_column('features')
        batch_op.drop_column('regime')
