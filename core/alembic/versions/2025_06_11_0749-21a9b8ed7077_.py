"""empty message

Revision ID: 21a9b8ed7077
Revises: 55c5dd3f5a82
Create Date: 2025-06-11 07:49:20.388140

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '21a9b8ed7077'
down_revision: Union[str, None] = '55c5dd3f5a82'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table('agent_trains',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.Integer(), nullable=False),
    sa.Column('epochs', sa.Integer(), nullable=False),
    sa.Column('batch_size', sa.Integer(), nullable=False),
    sa.Column('learning_rate', sa.Float(), nullable=False),
    sa.Column('weight_decay', sa.Float(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('created', sa.DateTime(), nullable=False),
    sa.Column('updated', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], name=op.f('fk_agent_trains_agent_id_agents')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_agent_trains_user_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_agent_trains'))
    )
    op.create_table('news_coins',
    sa.Column('news_id', sa.Integer(), nullable=False),
    sa.Column('coin_id', sa.Integer(), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('created', sa.DateTime(), nullable=False),
    sa.Column('updated', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['coin_id'], ['coins.id'], name=op.f('fk_news_coins_coin_id_coins')),
    sa.ForeignKeyConstraint(['news_id'], ['newss.id'], name=op.f('fk_news_coins_news_id_newss')),
    sa.PrimaryKeyConstraint('news_id', 'coin_id', name=op.f('pk_news_coins'))
    )
    op.create_table('news_history_coins',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('id_news', sa.Integer(), nullable=False),
    sa.Column('coin_id', sa.Integer(), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('news_score_global', sa.Integer(), nullable=False),
    sa.Column('created', sa.DateTime(), nullable=False),
    sa.Column('updated', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['coin_id'], ['coins.id'], name=op.f('fk_news_history_coins_coin_id_coins')),
    sa.ForeignKeyConstraint(['id_news'], ['newss.id'], name=op.f('fk_news_history_coins_id_news_newss')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_news_history_coins'))
    )
    op.create_table('strategys',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('type', sa.String(length=50), nullable=False),
    sa.Column('model_risk_id', sa.Integer(), nullable=True, comment="ID модели с типом 'RiskModel'"),
    sa.Column('model_order_id', sa.Integer(), nullable=True, comment="ID модели с типом 'OrderModel'"),
    sa.Column('risk', sa.Float(), nullable=False),
    sa.Column('reward', sa.Float(), nullable=False),
    sa.Column('created', sa.DateTime(), nullable=False),
    sa.Column('updated', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['model_order_id'], ['m_l__models.id'], name=op.f('fk_strategys_model_order_id_m_l__models')),
    sa.ForeignKeyConstraint(['model_risk_id'], ['m_l__models.id'], name=op.f('fk_strategys_model_risk_id_m_l__models')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_strategys_user_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_strategys'))
    )
    op.create_table('strategy_agents',
    sa.Column('strategy_id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.Integer(), nullable=False),
    sa.Column('created', sa.DateTime(), nullable=False),
    sa.Column('updated', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], name=op.f('fk_strategy_agents_agent_id_agents')),
    sa.ForeignKeyConstraint(['strategy_id'], ['strategys.id'], name=op.f('fk_strategy_agents_strategy_id_strategys')),
    sa.PrimaryKeyConstraint('strategy_id', 'agent_id', name=op.f('pk_strategy_agents'))
    )
    op.create_table('strategy_coins',
    sa.Column('strategy_id', sa.Integer(), nullable=False),
    sa.Column('coin_id', sa.Integer(), nullable=False),
    sa.Column('created', sa.DateTime(), nullable=False),
    sa.Column('updated', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['coin_id'], ['coins.id'], name=op.f('fk_strategy_coins_coin_id_coins')),
    sa.ForeignKeyConstraint(['strategy_id'], ['strategys.id'], name=op.f('fk_strategy_coins_strategy_id_strategys')),
    sa.PrimaryKeyConstraint('strategy_id', 'coin_id', name=op.f('pk_strategy_coins'))
    )
    op.add_column('agent_actions', sa.Column('agent_id', sa.Integer(), nullable=False))
    op.drop_constraint(op.f('fk_agent_actions_id_agent_agents'), 'agent_actions', type_='foreignkey')
    op.create_foreign_key(op.f('fk_agent_actions_agent_id_agents'), 'agent_actions', 'agents', ['agent_id'], ['id'])
    op.drop_column('agent_actions', 'id_agent')
    op.add_column('coins', sa.Column('news_score_global', sa.Integer(), nullable=True))
    op.execute("UPDATE coins SET news_score_global = 50")
    op.add_column('model_actions', sa.Column('model_id', sa.Integer(), nullable=False))
    op.drop_constraint(op.f('fk_model_actions_id_model_m_l__models'), 'model_actions', type_='foreignkey')
    op.create_foreign_key(op.f('fk_model_actions_model_id_m_l__models'), 'model_actions', 'm_l__models', ['model_id'], ['id'])
    op.drop_column('model_actions', 'id_model')
    op.add_column('statistic_agents', sa.Column('agent_id', sa.Integer(), nullable=False))
    op.drop_constraint(op.f('fk_statistic_agents_id_agnet_agents'), 'statistic_agents', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_agents_agent_id_agents'), 'statistic_agents', 'agents', ['agent_id'], ['id'])
    op.drop_column('statistic_agents', 'id_agnet')
    op.add_column('statistic_models', sa.Column('model_id', sa.Integer(), nullable=False))
    op.drop_constraint(op.f('fk_statistic_models_id_model_m_l__models'), 'statistic_models', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_models_model_id_m_l__models'), 'statistic_models', 'm_l__models', ['model_id'], ['id'])
    op.drop_column('statistic_models', 'id_model')
    op.drop_column('users', 'risk_score')


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('risk_score', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('statistic_models', sa.Column('id_model', sa.INTEGER(), autoincrement=False, nullable=False))
    op.drop_constraint(op.f('fk_statistic_models_model_id_m_l__models'), 'statistic_models', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_models_id_model_m_l__models'), 'statistic_models', 'm_l__models', ['id_model'], ['id'])
    op.drop_column('statistic_models', 'model_id')
    op.add_column('statistic_agents', sa.Column('id_agnet', sa.INTEGER(), autoincrement=False, nullable=False))
    op.drop_constraint(op.f('fk_statistic_agents_agent_id_agents'), 'statistic_agents', type_='foreignkey')
    op.create_foreign_key(op.f('fk_statistic_agents_id_agnet_agents'), 'statistic_agents', 'agents', ['id_agnet'], ['id'])
    op.drop_column('statistic_agents', 'agent_id')
    op.add_column('model_actions', sa.Column('id_model', sa.INTEGER(), autoincrement=False, nullable=False))
    op.drop_constraint(op.f('fk_model_actions_model_id_m_l__models'), 'model_actions', type_='foreignkey')
    op.create_foreign_key(op.f('fk_model_actions_id_model_m_l__models'), 'model_actions', 'm_l__models', ['id_model'], ['id'])
    op.drop_column('model_actions', 'model_id')
    op.drop_column('coins', 'news_score_global')
    op.add_column('agent_actions', sa.Column('id_agent', sa.INTEGER(), autoincrement=False, nullable=False))
    op.drop_constraint(op.f('fk_agent_actions_agent_id_agents'), 'agent_actions', type_='foreignkey')
    op.create_foreign_key(op.f('fk_agent_actions_id_agent_agents'), 'agent_actions', 'agents', ['id_agent'], ['id'])
    op.drop_column('agent_actions', 'agent_id')
    op.drop_table('strategy_coins')
    op.drop_table('strategy_agents')
    op.drop_table('strategys')
    op.drop_table('news_history_coins')
    op.drop_table('news_coins')
    op.drop_table('agent_trains')
    # ### end Alembic commands ###
