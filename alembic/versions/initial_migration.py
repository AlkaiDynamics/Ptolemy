"""Initial migration

Revision ID: initial_migration
Revises: 
Create Date: 2023-11-15

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = 'initial_migration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create events table
    op.create_table('events',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('type', sa.String(), nullable=True),
        sa.Column('data', sqlite.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_events_type'), 'events', ['type'], unique=False)
    
    # Create relationships table
    op.create_table('relationships',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('source_entity', sa.String(), nullable=True),
        sa.Column('target_entity', sa.String(), nullable=True),
        sa.Column('relationship_type', sa.String(), nullable=True),
        sa.Column('meta_data', sqlite.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_relationships_relationship_type'), 'relationships', ['relationship_type'], unique=False)
    op.create_index(op.f('ix_relationships_source_entity'), 'relationships', ['source_entity'], unique=False)
    op.create_index(op.f('ix_relationships_target_entity'), 'relationships', ['target_entity'], unique=False)
    
    # Create patterns table
    op.create_table('patterns',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('type', sa.String(), nullable=True),
        sa.Column('implementation', sa.String(), nullable=True),
        sa.Column('meta_data', sqlite.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_patterns_name'), 'patterns', ['name'], unique=False)
    op.create_index(op.f('ix_patterns_type'), 'patterns', ['type'], unique=False)
    
    # Create insights table
    op.create_table('insights',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=True),
        sa.Column('content', sa.String(), nullable=True),
        sa.Column('relevance', sa.Float(), nullable=True),
        sa.Column('meta_data', sqlite.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_insights_type'), 'insights', ['type'], unique=False)


def downgrade() -> None:
    op.drop_table('insights')
    op.drop_table('patterns')
    op.drop_table('relationships')
    op.drop_table('events')
