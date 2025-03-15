#!/usr/bin/env python
# tests/latent_reasoning/test_engine.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from ptolemy.context_engine import ContextEngine
from ptolemy.latent_reasoning.engine import LatentReasoningEngine
from ptolemy.latent_reasoning.state import LatentState


@pytest.fixture
def mock_context_engine():
    """Create a mock ContextEngine for testing."""
    context_engine = MagicMock(spec=ContextEngine)
    context_engine.get_model_context.return_value = "Test context with relevant information"
    return context_engine


@pytest.fixture
def latent_reasoning_engine(mock_context_engine):
    """Create a LatentReasoningEngine instance for testing."""
    return LatentReasoningEngine(
        context_engine=mock_context_engine,
        hidden_dim=128,
        default_iterations=4,
        max_iterations=24,
        convergence_threshold=0.05  # Higher threshold for testing
    )


@pytest.mark.asyncio
async def test_process_basic(latent_reasoning_engine):
    """Test the basic processing functionality."""
    # Arrange
    task = "Test task"
    
    # Act
    result = await latent_reasoning_engine.process(task, iterations=4)
    
    # Assert
    assert "process_id" in result
    assert "output" in result
    assert "iterations" in result
    assert result["iterations"] <= 4
    assert "time" in result
    assert result["time"] > 0
    
    # Check output structure
    output = result["output"]
    assert "context" in output
    assert "key_concepts" in output
    assert isinstance(output["key_concepts"], list)
    assert "attention_focus" in output


@pytest.mark.asyncio
async def test_process_with_adaptive_stopping(latent_reasoning_engine):
    """Test that adaptive stopping works correctly."""
    # We need to mock the behavior to ensure adaptive stopping
    # by controlling the convergence pattern
    
    # Create a patched version of calculate_change that returns decreasing values
    original_calculate_change = LatentState.calculate_change
    
    def mock_calculate_change(self, other_state):
        # Return a decreasing value based on iteration number to trigger convergence
        if self.iteration > 2:
            return 0.01  # Value below the threshold to trigger stopping
        return 0.5  # Higher value for initial iterations
    
    # Patch the method
    with patch.object(LatentState, 'calculate_change', mock_calculate_change):
        # Act
        result = await latent_reasoning_engine.process(
            "Test task for adaptive stopping",
            iterations=32,  # Large number to ensure adaptive stopping has a chance to kick in
            adaptive=True
        )
    
    # Assert
    assert result["iterations"] < 32
    assert result["adaptive_stopped"] is True


@pytest.mark.asyncio
async def test_process_with_trajectory(latent_reasoning_engine):
    """Test that trajectory information is returned when requested."""
    # Arrange
    task = "Test task with trajectory"
    task_metadata = {"include_trajectory": True}
    
    # Act
    result = await latent_reasoning_engine.process(
        task, 
        iterations=4,
        task_metadata=task_metadata
    )
    
    # Assert
    assert "state_trajectory" in result
    trajectory = result["state_trajectory"]
    assert "steps" in trajectory
    assert trajectory["steps"] == result["iterations"] + 1  # +1 for initial state
    assert "convergence_pattern" in trajectory
    assert len(trajectory["convergence_pattern"]) == result["iterations"]


@pytest.mark.asyncio
async def test_process_with_error_handling(latent_reasoning_engine, mock_context_engine):
    """Test error handling during processing."""
    # Arrange
    task = "Test task for error handling"
    mock_context_engine.get_model_context.side_effect = Exception("Test error")
    
    # Act
    result = await latent_reasoning_engine.process(task)
    
    # Assert
    assert "error" in result
    assert "Test error" in result["error"]
    assert result["iterations"] == 0


@pytest.mark.asyncio
async def test_analyze_task_complexity(latent_reasoning_engine):
    """Test task complexity analysis."""
    # Arrange
    simple_task = "Simple task"
    complex_task = "Complex task with multiple steps. First, do this? Then, do that? " + \
                  "Finally implement the following code: ```python\nprint('hello')\n```"
    
    # Act
    simple_result = await latent_reasoning_engine.analyze_task_complexity(simple_task, "Short context")
    complex_result = await latent_reasoning_engine.analyze_task_complexity(complex_task, "Long context " * 50)
    
    # Assert
    assert "complexity_score" in simple_result
    assert "recommended_iterations" in simple_result
    assert "indicators" in simple_result
    
    assert simple_result["complexity_score"] < complex_result["complexity_score"]
    assert simple_result["recommended_iterations"] <= complex_result["recommended_iterations"]


@pytest.mark.asyncio
async def test_metrics_recording(latent_reasoning_engine):
    """Test that metrics are properly recorded."""
    # Arrange
    task = "Test task for metrics"
    
    # Act
    await latent_reasoning_engine.process(task, iterations=4)
    metrics = latent_reasoning_engine.get_metrics_summary()
    
    # Assert
    assert metrics["total_processes"] == 1
    assert metrics["total_iterations"] > 0
    assert metrics["avg_iterations_per_process"] > 0
    assert "task_types" in metrics
