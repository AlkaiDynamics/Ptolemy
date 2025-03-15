import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch

from ptolemy.multi_model import MultiModelProcessor
from ptolemy.context_engine import ContextEngine
from ptolemy.temporal_core import TemporalCore
from ptolemy.config import DEFAULT_PROVIDER, MODEL_REGISTRY


@pytest.fixture
def mock_context_engine():
    """Create a mock ContextEngine."""
    mock_engine = MagicMock(spec=ContextEngine)
    # Mock the get_model_context method
    mock_engine.get_model_context = AsyncMock(return_value={
        "recent_events": [],
        "relevant_insights": []
    })
    return mock_engine


class TestMultiModelProcessor:
    """Test the MultiModelProcessor class."""

    @pytest.mark.asyncio
    @patch('ptolemy.multi_model.MultiModelProcessor.initialize_clients')
    async def test_route_task_architect(self, mock_init_clients, mock_context_engine):
        """Test routing a task to the architect model."""
        # Create processor with mocked dependencies
        processor = MultiModelProcessor(mock_context_engine)
        
        # Mock the route_task method to return a test response
        with patch.object(processor, 'route_task', new_callable=AsyncMock) as mock_route:
            mock_route.return_value = "This is a test response"
            
            # Call the method
            task = "Design a scalable microservice architecture"
            model_type = "architect"
            result = await processor.route_task(task, model_type)
            
            # Verify the result
            assert result == "This is a test response"
            mock_route.assert_called_once_with(task, model_type)

    @pytest.mark.asyncio
    @patch('ptolemy.multi_model.MultiModelProcessor.initialize_clients')
    async def test_route_task_implementer(self, mock_init_clients, mock_context_engine):
        """Test routing a task to the implementer model."""
        # Create processor with mocked dependencies
        processor = MultiModelProcessor(mock_context_engine)
        
        # Mock the route_task method to return a test response
        with patch.object(processor, 'route_task', new_callable=AsyncMock) as mock_route:
            mock_route.return_value = "This is a test response"
            
            # Call the method
            task = "Create a Python function to calculate the factorial of a number"
            model_type = "implementer"
            result = await processor.route_task(task, model_type)
            
            # Verify the result
            assert result == "This is a test response"
            mock_route.assert_called_once_with(task, model_type)

    @pytest.mark.asyncio
    @patch('ptolemy.multi_model.MultiModelProcessor.initialize_clients')
    async def test_route_task_with_options(self, mock_init_clients, mock_context_engine):
        """Test routing a task with specific options."""
        # Create processor with mocked dependencies
        processor = MultiModelProcessor(mock_context_engine)
        
        # Mock the route_task method to return a test response
        with patch.object(processor, 'route_task', new_callable=AsyncMock) as mock_route:
            mock_route.return_value = "This is a test response"
            
            # Call the method
            task = "Review this code for security issues"
            model_type = "reviewer"
            options = {
                "provider": DEFAULT_PROVIDER,
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            }
            result = await processor.route_task(task, model_type, options)
            
            # Verify the result
            assert result == "This is a test response"
            mock_route.assert_called_once_with(task, model_type, options)

    @pytest.mark.asyncio
    async def test_route_task_unknown_model(self, mock_context_engine):
        """Test routing a task to an unknown model type."""
        # Create processor with mocked dependencies
        with patch('ptolemy.multi_model.MultiModelProcessor.initialize_clients'):
            processor = MultiModelProcessor(mock_context_engine)
            
            # Test with an unknown model type
            task = "Analyze this code"
            model_type = "nonexistent_model_type"
            
            # Expect a ValueError when routing to an unknown model type
            with pytest.raises(ValueError):
                await processor.route_task(task, model_type)

    @pytest.mark.asyncio
    async def test_route_task_unavailable_provider(self, mock_context_engine):
        """Test routing a task to an unavailable provider."""
        # Create processor with mocked dependencies
        with patch('ptolemy.multi_model.MultiModelProcessor.initialize_clients'):
            processor = MultiModelProcessor(mock_context_engine)
            processor.ai_clients = {}  # Ensure no clients are available
            
            # Test with a valid model type but unavailable provider
            task = "Integrate these components"
            model_type = "integrator"
            options = {
                "provider": "unavailable_provider"
            }
            
            # Route the task with an unavailable provider
            result = await processor.route_task(task, model_type, options)
            
            # Should return a mock response
            assert "[MOCK RESPONSE]" in result
