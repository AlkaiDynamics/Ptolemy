import os
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ptolemy.temporal_core import TemporalCore
from ptolemy.context_engine import ContextEngine
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.feedback import FeedbackOrchestrator
from ptolemy.config import DEFAULT_PROVIDER


# Helper function to create an awaitable mock
def async_return(result):
    """Create an awaitable mock that returns the given result."""
    async_mock = AsyncMock()
    async_mock.return_value = result
    return async_mock


@pytest.fixture
def temporal_core():
    """Create a mocked TemporalCore instance."""
    # Create a mock TemporalCore
    core = MagicMock(spec=TemporalCore)
    
    # Mock the async methods
    core.record_event = AsyncMock(return_value="mock_event_id")
    core.get_events = AsyncMock(return_value=[{
        "id": "mock_event_id",
        "type": "code_generation",
        "content": "Generated a function to calculate Fibonacci numbers",
        "metadata": {"language": "python", "complexity": "medium"},
        "timestamp": "2025-03-15T12:00:00Z"
    }])
    
    return core


@pytest.fixture
def context_engine(temporal_core):
    """Create a mocked ContextEngine instance."""
    # Create a mock ContextEngine
    engine = MagicMock(spec=ContextEngine)
    
    # Mock the async methods
    engine.store_insight = AsyncMock(return_value="mock_insight_id")
    engine.get_insights = AsyncMock(return_value=[{
        "id": "mock_insight_id",
        "category": "performance",
        "content": "Use list comprehensions instead of map() for better readability",
        "confidence": 0.85,
        "metadata": {"source": "best_practices"},
        "timestamp": "2025-03-15T12:00:00Z"
    }])
    engine.get_model_context = AsyncMock(return_value={
        "recent_events": [],
        "relevant_insights": []
    })
    
    # Set the temporal_core attribute
    engine.temporal_core = temporal_core
    
    return engine


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    # Create a mock client
    mock_client = MagicMock()
    
    # Create a mock for the chat completions create method
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "This is a test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    # Create an async method for chat.completions.create
    async def mock_create(*args, **kwargs):
        return mock_response
    
    # Set up the mock client structure
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = mock_create
    
    return mock_client


@pytest.fixture
def multi_model(context_engine, mock_openai_client):
    """Create a mocked MultiModelProcessor instance."""
    # Create a mock MultiModelProcessor
    processor = MagicMock(spec=MultiModelProcessor)
    
    # Set the context_engine attribute
    processor.context_engine = context_engine
    
    # Set up the ai_clients dictionary
    processor.ai_clients = {DEFAULT_PROVIDER: mock_openai_client}
    
    # Mock the route_task method
    async def mock_route_task(task, model_type, **kwargs):
        return "This is a test response"
    
    processor.route_task = mock_route_task
    
    return processor


@pytest.fixture
def feedback_orchestrator(temporal_core, context_engine):
    """Create a mocked FeedbackOrchestrator instance."""
    # Create a mock FeedbackOrchestrator
    orchestrator = MagicMock(spec=FeedbackOrchestrator)
    
    # Set the attributes
    orchestrator.temporal_core = temporal_core
    orchestrator.context_engine = context_engine
    
    # Mock the record_feedback method
    async def mock_record_feedback(content, rating, metadata=None):
        return "mock_event_id"
    
    orchestrator.record_feedback = mock_record_feedback
    
    return orchestrator


@pytest.mark.asyncio
async def test_record_event(temporal_core):
    """Test recording an event."""
    event_data = {
        "type": "code_generation",
        "content": "Generated a function to calculate Fibonacci numbers",
        "metadata": {"language": "python", "complexity": "medium"}
    }
    
    event_id = await temporal_core.record_event(
        event_data["type"],
        event_data["content"],
        event_data["metadata"]
    )
    
    assert event_id is not None
    assert isinstance(event_id, str)
    assert event_id == "mock_event_id"
    
    # Verify the event was recorded
    events = await temporal_core.get_events(limit=10)
    assert len(events) > 0
    
    # Find our event
    found = False
    for event in events:
        if event["id"] == event_id:
            found = True
            assert event["type"] == event_data["type"]
            assert event["content"] == event_data["content"]
            assert event["metadata"] == event_data["metadata"]
    
    assert found, "Event not found in retrieved events"


@pytest.mark.asyncio
async def test_store_insight(context_engine):
    """Test storing an insight."""
    insight_data = {
        "category": "code_style",
        "content": "Use descriptive variable names",
        "confidence": 0.9,
        "metadata": {"source": "static_analysis"}
    }
    
    insight_id = await context_engine.store_insight(
        insight_data["category"],
        insight_data["content"],
        insight_data["confidence"],
        insight_data["metadata"]
    )
    
    assert insight_id is not None
    assert isinstance(insight_id, str)
    assert insight_id == "mock_insight_id"


@pytest.mark.asyncio
async def test_get_insights(context_engine):
    """Test retrieving insights."""
    # Store a test insight first
    await context_engine.store_insight(
        "performance",
        "Use list comprehensions instead of map() for better readability",
        0.85,
        {"source": "best_practices"}
    )
    
    # Retrieve insights
    insights = await context_engine.get_insights(category="performance")
    
    assert len(insights) > 0
    
    # Verify the insight data
    insight = insights[0]
    assert insight["category"] == "performance"
    assert "list comprehensions" in insight["content"]
    assert insight["confidence"] == 0.85
    assert insight["metadata"]["source"] == "best_practices"


@pytest.mark.asyncio
async def test_record_feedback(feedback_orchestrator):
    """Test recording feedback."""
    feedback_data = {
        "content": "The generated code is efficient but could use more comments",
        "rating": 4,
        "metadata": {"source": "user_interface"}
    }
    
    feedback_id = await feedback_orchestrator.record_feedback(
        feedback_data["content"],
        feedback_data["rating"],
        feedback_data["metadata"]
    )
    
    assert feedback_id is not None
    assert isinstance(feedback_id, str)
    assert feedback_id == "mock_event_id"


@pytest.mark.asyncio
async def test_route_task(multi_model):
    """Test routing a task to a model."""
    task = "Create a Python function to calculate the factorial of a number"
    model_type = "implementer"
    
    # Route the task
    result = await multi_model.route_task(task, model_type)
    
    # Verify the result
    assert result is not None
    assert isinstance(result, str)
    assert result == "This is a test response"
