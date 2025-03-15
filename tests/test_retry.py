import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from ptolemy.utils import async_retry

@pytest.mark.asyncio
async def test_successful_function_no_retry():
    """Test that successful functions execute only once."""
    mock_func = AsyncMock(return_value="success")
    
    # Apply the decorator
    decorated = async_retry()(mock_func)
    
    # Call the decorated function
    result = await decorated("arg1", key="value")
    
    # Check results
    assert result == "success"
    mock_func.assert_called_once_with("arg1", key="value")

@pytest.mark.asyncio
async def test_function_with_retries():
    """Test that failing functions retry the specified number of times."""
    # Create a mock that fails twice then succeeds
    mock_func = AsyncMock(side_effect=[
        Exception("First failure"),
        Exception("Second failure"),
        "success"
    ])
    
    # Apply the decorator with minimal delay for testing
    decorated = async_retry(max_retries=3, base_delay=0.01)(mock_func)
    
    # Call the decorated function
    result = await decorated()
    
    # Check results
    assert result == "success"
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_function_exceeds_max_retries():
    """Test that exceeding max retries raises the original exception."""
    # Create a mock that always fails
    mock_func = AsyncMock(side_effect=Exception("Always fails"))
    
    # Apply the decorator with minimal delay for testing
    decorated = async_retry(max_retries=2, base_delay=0.01)(mock_func)
    
    # Call the decorated function
    with pytest.raises(Exception, match="Always fails"):
        await decorated()
    
    # Check that it was called the expected number of times
    assert mock_func.call_count == 3  # Initial call + 2 retries
