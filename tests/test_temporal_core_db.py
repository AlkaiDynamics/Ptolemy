import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
import os
import uuid

from ptolemy.temporal_core import TemporalCore
from ptolemy.database import async_session, Event, init_db
from sqlalchemy import select, func, text

@pytest_asyncio.fixture
async def init_test_db():
    """Initialize test database."""
    await init_db()
    async with async_session() as session:
        # Clear any existing test data
        await session.execute(text("DELETE FROM events"))
        await session.commit()

@pytest.mark.asyncio
async def test_record_event_db_storage(init_test_db):
    """Test that events are stored in both filesystem and database."""
    # Create a temporal core with db enabled
    core = TemporalCore()
    core.db_enabled = True
    await core.initialize()
    
    # Record an event
    event_data = {"test_key": "test_value"}
    event = await core.record_event("test_event", event_data)
    
    # Check that the event was saved to the database
    async with async_session() as session:
        result = await session.execute(select(Event).where(Event.id == event["id"]))
        db_event = result.scalars().first()
        
        assert db_event is not None
        assert db_event.id == event["id"]
        assert db_event.type == "test_event"
        assert db_event.data == event_data

@pytest.mark.asyncio
async def test_get_events_from_db(init_test_db):
    """Test retrieving events from the database with filters."""
    # Create a temporal core with db enabled
    core = TemporalCore()
    core.db_enabled = True
    await core.initialize()
    
    # Clear any existing events first
    async with async_session() as session:
        await session.execute(text("DELETE FROM events"))
        await session.commit()
    
    # Record multiple events
    await core.record_event("event_type_1", {"order": 1})
    await core.record_event("event_type_2", {"order": 2})
    await core.record_event("event_type_1", {"order": 3})
    
    # Get events with type filter
    events = await core.get_events({"type": "event_type_1"})
    
    assert len(events) == 2
    assert all(e["type"] == "event_type_1" for e in events)
    assert events[0]["data"]["order"] < events[1]["data"]["order"]  # Check ordering

@pytest.mark.asyncio
async def test_get_event_by_id_from_db(init_test_db):
    """Test retrieving a specific event by ID from the database."""
    # Create a temporal core with db enabled
    core = TemporalCore()
    core.db_enabled = True
    await core.initialize()
    
    # Record an event
    original = await core.record_event("test_event", {"test_key": "test_value"})
    
    # Get the event by ID
    retrieved = await core.get_event_by_id(original["id"])
    
    assert retrieved["id"] == original["id"]
    assert retrieved["type"] == original["type"]
    assert retrieved["data"] == original["data"]

@pytest.mark.asyncio
async def test_fallback_to_file_storage(init_test_db):
    """Test fallback to file storage when database is disabled."""
    # Create a temporal core with db disabled
    core = TemporalCore()
    core.db_enabled = False
    await core.initialize()
    
    # Mock the file storage methods
    with patch.object(core, '_save_event_to_file', AsyncMock()) as mock_save:
        # Record an event
        event = await core.record_event("test_event", {"test_key": "test_value"})
        
        # Check that file storage was used
        mock_save.assert_called_once()
        
        # Check that the event was not saved to the database
        async with async_session() as session:
            result = await session.execute(select(Event).where(Event.id == event["id"]))
            db_event = result.scalars().first()
            assert db_event is None

@pytest.mark.asyncio
async def test_db_and_file_storage_integration(init_test_db):
    """Test that both database and file storage work together."""
    # Create a temporal core with db enabled
    core = TemporalCore()
    core.db_enabled = True
    await core.initialize()
    
    # Mock the file storage methods to verify they're called
    with patch.object(core, '_save_event_to_file', AsyncMock()) as mock_save:
        # Record an event
        event = await core.record_event("test_event", {"test_key": "test_value"})
        
        # Check that file storage was used
        mock_save.assert_called_once()
        
        # Check that the event was also saved to the database
        async with async_session() as session:
            result = await session.execute(select(Event).where(Event.id == event["id"]))
            db_event = result.scalars().first()
            assert db_event is not None
            assert db_event.id == event["id"]
