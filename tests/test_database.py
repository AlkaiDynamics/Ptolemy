import pytest
import pytest_asyncio
import uuid
from datetime import datetime
from sqlalchemy import select, text

from ptolemy.database import async_session, init_db, Event, Relationship, Pattern, Insight

@pytest_asyncio.fixture
async def init_test_db():
    """Initialize test database."""
    await init_db()
    async with async_session() as session:
        # Clear any existing test data
        await session.execute(text("DELETE FROM events"))
        await session.execute(text("DELETE FROM relationships"))
        await session.execute(text("DELETE FROM patterns"))
        await session.execute(text("DELETE FROM insights"))
        await session.commit()

@pytest.mark.asyncio
async def test_event_crud(init_test_db):
    """Test Create, Read, Update, Delete operations for Event model."""
    event_id = str(uuid.uuid4())
    
    # Create
    async with async_session() as session:
        event = Event(
            id=event_id,
            type="test_event",
            data={"key": "value"}
        )
        session.add(event)
        await session.commit()
    
    # Read
    async with async_session() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        fetched = result.scalars().first()
        
        assert fetched is not None
        assert fetched.id == event_id
        assert fetched.type == "test_event"
        assert fetched.data == {"key": "value"}
    
    # Update
    async with async_session() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        fetched = result.scalars().first()
        fetched.data = {"key": "updated"}
        await session.commit()
    
    # Verify update
    async with async_session() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        fetched = result.scalars().first()
        assert fetched.data == {"key": "updated"}
    
    # Delete
    async with async_session() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        fetched = result.scalars().first()
        await session.delete(fetched)
        await session.commit()
    
    # Verify deletion
    async with async_session() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        fetched = result.scalars().first()
        assert fetched is None

@pytest.mark.asyncio
async def test_relationship_crud(init_test_db):
    """Test CRUD operations for Relationship model."""
    relationship_id = str(uuid.uuid4())
    
    # Create
    async with async_session() as session:
        relationship = Relationship(
            id=relationship_id,
            source_entity="entity1",
            target_entity="entity2",
            relationship_type="test_relationship",
            meta_data={"key": "value"}
        )
        session.add(relationship)
        await session.commit()
    
    # Read
    async with async_session() as session:
        result = await session.execute(select(Relationship).where(Relationship.id == relationship_id))
        fetched = result.scalars().first()
        
        assert fetched is not None
        assert fetched.id == relationship_id
        assert fetched.source_entity == "entity1"
        assert fetched.target_entity == "entity2"
        assert fetched.relationship_type == "test_relationship"
        assert fetched.meta_data == {"key": "value"}
    
    # Update
    async with async_session() as session:
        result = await session.execute(select(Relationship).where(Relationship.id == relationship_id))
        fetched = result.scalars().first()
        fetched.meta_data = {"key": "updated"}
        await session.commit()
    
    # Verify update
    async with async_session() as session:
        result = await session.execute(select(Relationship).where(Relationship.id == relationship_id))
        fetched = result.scalars().first()
        assert fetched.meta_data == {"key": "updated"}
    
    # Delete
    async with async_session() as session:
        result = await session.execute(select(Relationship).where(Relationship.id == relationship_id))
        fetched = result.scalars().first()
        await session.delete(fetched)
        await session.commit()
    
    # Verify deletion
    async with async_session() as session:
        result = await session.execute(select(Relationship).where(Relationship.id == relationship_id))
        fetched = result.scalars().first()
        assert fetched is None

@pytest.mark.asyncio
async def test_pattern_crud(init_test_db):
    """Test CRUD operations for Pattern model."""
    pattern_id = str(uuid.uuid4())
    
    # Create
    async with async_session() as session:
        pattern = Pattern(
            id=pattern_id,
            name="test_pattern",
            type="python",
            implementation="def test(): pass",
            meta_data={"key": "value"}
        )
        session.add(pattern)
        await session.commit()
    
    # Read
    async with async_session() as session:
        result = await session.execute(select(Pattern).where(Pattern.id == pattern_id))
        fetched = result.scalars().first()
        
        assert fetched is not None
        assert fetched.id == pattern_id
        assert fetched.name == "test_pattern"
        assert fetched.type == "python"
        assert fetched.implementation == "def test(): pass"
        assert fetched.meta_data == {"key": "value"}
    
    # Update
    async with async_session() as session:
        result = await session.execute(select(Pattern).where(Pattern.id == pattern_id))
        fetched = result.scalars().first()
        fetched.implementation = "def updated_test(): pass"
        await session.commit()
    
    # Verify update
    async with async_session() as session:
        result = await session.execute(select(Pattern).where(Pattern.id == pattern_id))
        fetched = result.scalars().first()
        assert fetched.implementation == "def updated_test(): pass"
    
    # Delete
    async with async_session() as session:
        result = await session.execute(select(Pattern).where(Pattern.id == pattern_id))
        fetched = result.scalars().first()
        await session.delete(fetched)
        await session.commit()
    
    # Verify deletion
    async with async_session() as session:
        result = await session.execute(select(Pattern).where(Pattern.id == pattern_id))
        fetched = result.scalars().first()
        assert fetched is None

@pytest.mark.asyncio
async def test_insight_crud(init_test_db):
    """Test CRUD operations for Insight model."""
    insight_id = str(uuid.uuid4())
    
    # Create
    async with async_session() as session:
        insight = Insight(
            id=insight_id,
            type="test_insight",
            content="This is a test insight",
            relevance=0.85,
            meta_data={"key": "value"}
        )
        session.add(insight)
        await session.commit()
    
    # Read
    async with async_session() as session:
        result = await session.execute(select(Insight).where(Insight.id == insight_id))
        fetched = result.scalars().first()
        
        assert fetched is not None
        assert fetched.id == insight_id
        assert fetched.type == "test_insight"
        assert fetched.content == "This is a test insight"
        assert fetched.relevance == 0.85
        assert fetched.meta_data == {"key": "value"}
    
    # Update
    async with async_session() as session:
        result = await session.execute(select(Insight).where(Insight.id == insight_id))
        fetched = result.scalars().first()
        fetched.relevance = 0.95
        await session.commit()
    
    # Verify update
    async with async_session() as session:
        result = await session.execute(select(Insight).where(Insight.id == insight_id))
        fetched = result.scalars().first()
        assert fetched.relevance == 0.95
    
    # Delete
    async with async_session() as session:
        result = await session.execute(select(Insight).where(Insight.id == insight_id))
        fetched = result.scalars().first()
        await session.delete(fetched)
        await session.commit()
    
    # Verify deletion
    async with async_session() as session:
        result = await session.execute(select(Insight).where(Insight.id == insight_id))
        fetched = result.scalars().first()
        assert fetched is None
