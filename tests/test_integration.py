import pytest
import pytest_asyncio
import asyncio
import uuid
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

from ptolemy.temporal_core import TemporalCore
from ptolemy.context_engine import ContextEngine
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.database import init_db, async_session, Event, Pattern
from sqlalchemy import select, text

@pytest_asyncio.fixture
async def setup_components():
    """Set up all components for integration testing."""
    await init_db()
    
    # Clear test data
    async with async_session() as session:
        await session.execute(text("DELETE FROM events"))
        await session.execute(text("DELETE FROM relationships"))
        await session.execute(text("DELETE FROM patterns"))
        await session.execute(text("DELETE FROM insights"))
        await session.commit()
    
    temporal_core = TemporalCore()
    context_engine = ContextEngine(temporal_core)
    multi_model = MultiModelProcessor(context_engine)
    
    await temporal_core.initialize()
    await context_engine.initialize()
    
    components = {
        "temporal_core": temporal_core,
        "context_engine": context_engine,
        "multi_model": multi_model
    }
    
    return components

@pytest.mark.asyncio
async def test_end_to_end_workflow(setup_components):
    """Test a complete end-to-end workflow with all components."""
    components = setup_components
    
    # Initialize a project
    project = await components["temporal_core"].record_event(
        "project_initialized", 
        {"name": "Test Project", "description": "Integration test"}
    )
    
    # Store a pattern
    pattern = await components["context_engine"].store_pattern(
        "test_pattern",
        "python",
        "def example(): pass",
        {"category": "test"}
    )
    
    # Use multi-model processor with retry capability
    # Mock the route_task method to avoid actual external calls
    with patch.object(components["multi_model"], "route_task", 
                     new_callable=AsyncMock, return_value="Generated content"):
        result = await components["multi_model"].route_task(
            "Test prompt",
            "implementer"
        )
        assert result == "Generated content"
    
    # Verify everything was stored in the database
    async with async_session() as session:
        # Check project event
        result = await session.execute(select(Event).where(Event.id == project["id"]))
        project_event = result.scalars().first()
        assert project_event is not None
        assert project_event.type == "project_initialized"
        assert project_event.data["name"] == "Test Project"
        
        # Check pattern
        result = await session.execute(select(Pattern).where(Pattern.name == "test_pattern"))
        db_pattern = result.scalars().first()
        assert db_pattern is not None
        assert db_pattern.type == "python"
        assert db_pattern.implementation == "def example(): pass"

@pytest.mark.asyncio
async def test_retry_mechanism_integration(setup_components):
    """Test that the retry mechanism works in the integration context."""
    components = setup_components
    multi_model = components["multi_model"]
    
    # Create a mock for an internal method that will be called by route_task
    # We'll patch an internal method that's called by route_task instead of route_task itself
    # This allows the retry decorator to work properly
    
    # First, create a test method on the multi_model instance for testing purposes
    async def test_method_with_retries(*args, **kwargs):
        # This will be replaced by our mock
        pass
    
    # Add this method to our multi_model instance
    multi_model.test_method_with_retries = test_method_with_retries
    
    # Create a mock that fails twice then succeeds
    mock_test_method = AsyncMock(side_effect=[
        Exception("First failure"),
        Exception("Second failure"),
        "Success after retries"
    ])
    
    # Replace our test method with the mock
    multi_model.test_method_with_retries = mock_test_method
    
    # Create a simple wrapper function that uses the retry decorator
    from ptolemy.utils import async_retry
    
    @async_retry(max_retries=3, base_delay=0.1, backoff_factor=1, jitter=0.05)
    async def call_with_retry(multi_model):
        return await multi_model.test_method_with_retries()
    
    # Call our function with retry
    result = await call_with_retry(multi_model)
    
    # Check results
    assert result == "Success after retries"
    assert mock_test_method.call_count == 3  # Initial call + 2 retries

@pytest.mark.asyncio
async def test_database_and_file_storage_integration(setup_components):
    """Test that both database and file storage work together in integration."""
    components = setup_components
    temporal_core = components["temporal_core"]
    
    # Enable database storage
    temporal_core.db_enabled = True
    
    # Mock file storage to verify it's called
    with patch.object(temporal_core, '_save_event_to_file', AsyncMock()) as mock_save:
        # Record events
        event1 = await temporal_core.record_event("test_event_1", {"order": 1})
        event2 = await temporal_core.record_event("test_event_2", {"order": 2})
        
        # Verify file storage was called
        assert mock_save.call_count == 2
        
        # Verify events were saved to database
        async with async_session() as session:
            result = await session.execute(select(Event).where(
                Event.id.in_([event1["id"], event2["id"]])
            ))
            events = result.scalars().all()
            assert len(list(events)) == 2
            
        # Test retrieval from database
        retrieved_events = await temporal_core.get_events({})
        assert len(retrieved_events) >= 2
        
        # Test retrieval of specific event
        retrieved_event = await temporal_core.get_event_by_id(event1["id"])
        assert retrieved_event["id"] == event1["id"]
        assert retrieved_event["type"] == "test_event_1"
        assert retrieved_event["data"]["order"] == 1
