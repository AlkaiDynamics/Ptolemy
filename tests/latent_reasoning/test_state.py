#!/usr/bin/env python
# tests/latent_reasoning/test_state.py
import pytest
import numpy as np
from ptolemy.latent_reasoning.state import LatentState


@pytest.fixture
def sample_state():
    """Create a sample LatentState for testing."""
    # Create a random hidden state
    np.random.seed(42)  # For reproducibility
    hidden_state = np.random.randn(128) * 0.01
    
    # Create a sample state
    state = LatentState(hidden_state=hidden_state)
    
    # Add some test data
    state.context_embeddings = {
        "context_1": np.random.randn(128) * 0.01,
        "context_2": np.random.randn(128) * 0.01
    }
    state.task_embedding = np.random.randn(128) * 0.01
    state.iteration = 5
    state.add_key_concept("test_concept_1")
    state.add_key_concept("test_concept_2")
    state.update_attention("context_1", 0.7)
    state.update_attention("context_2", 0.3)
    state.add_reasoning_step("Initial step")
    state.add_reasoning_step("Processing step")
    
    return state


def test_state_initialization():
    """Test basic state initialization."""
    # Arrange
    hidden_state = np.ones(10)
    
    # Act
    state = LatentState(hidden_state=hidden_state)
    
    # Assert
    assert np.array_equal(state.hidden_state, hidden_state)
    assert state.context_embeddings == {}
    assert state.task_embedding is None
    assert state.iteration == 0
    assert len(state.key_concepts) == 0
    assert len(state.attention_weights) == 0
    assert len(state.reasoning_path) == 0


def test_copy_state(sample_state):
    """Test that state is properly copied."""
    # Act
    copied_state = sample_state.copy_state()
    
    # Assert - verify they're equal but not the same object
    assert copied_state is not sample_state
    assert copied_state.iteration == sample_state.iteration
    assert copied_state.key_concepts == sample_state.key_concepts
    assert np.array_equal(copied_state.hidden_state, sample_state.hidden_state)
    
    # Modify original and verify copy is unchanged
    sample_state.add_key_concept("new_concept")
    assert "new_concept" not in copied_state.key_concepts


def test_update_hidden_state(sample_state):
    """Test updating the hidden state."""
    # Arrange
    initial_norm_count = len(sample_state.prev_norms)
    new_hidden_state = np.zeros(128)
    
    # Act
    sample_state.update_hidden_state(new_hidden_state)
    
    # Assert
    assert np.array_equal(sample_state.hidden_state, new_hidden_state)
    assert len(sample_state.prev_norms) == initial_norm_count + 1
    assert sample_state.prev_norms[-1] == 0.0  # Norm of zero vector


def test_add_key_concept(sample_state):
    """Test adding key concepts."""
    # Arrange
    initial_count = len(sample_state.key_concepts)
    
    # Act
    sample_state.add_key_concept("new_concept")
    
    # Assert
    assert len(sample_state.key_concepts) == initial_count + 1
    assert "new_concept" in sample_state.key_concepts
    
    # Test adding duplicate concept doesn't increase count (set behavior)
    sample_state.add_key_concept("new_concept")
    assert len(sample_state.key_concepts) == initial_count + 1


def test_update_attention(sample_state):
    """Test updating attention weights."""
    # Act
    sample_state.update_attention("test_key", 0.5)
    
    # Assert
    assert "test_key" in sample_state.attention_weights
    assert sample_state.attention_weights["test_key"] == 0.5
    
    # Test updating existing key
    sample_state.update_attention("test_key", 0.8)
    assert sample_state.attention_weights["test_key"] == 0.8


def test_add_reasoning_step(sample_state):
    """Test adding reasoning steps."""
    # Arrange
    initial_count = len(sample_state.reasoning_path)
    
    # Act
    sample_state.add_reasoning_step("New step")
    
    # Assert
    assert len(sample_state.reasoning_path) == initial_count + 1
    assert sample_state.reasoning_path[-1] == "New step"


def test_calculate_change():
    """Test calculating change between states."""
    # Arrange
    state1 = LatentState(hidden_state=np.array([1.0, 0.0, 0.0]))
    state2 = LatentState(hidden_state=np.array([0.0, 1.0, 0.0]))
    state3 = LatentState(hidden_state=np.array([0.5, 0.5, 0.0]))
    
    # Act & Assert
    # Orthogonal vectors should have maximum change (1.0)
    assert state1.calculate_change(state2) == 1.0
    
    # Same vector should have no change (0.0)
    assert state1.calculate_change(state1) == 0.0
    
    # Partially similar vectors should have intermediate change
    change = state1.calculate_change(state3)
    assert 0.0 < change < 1.0
    
    # Null case
    assert state1.calculate_change(None) == 1.0


def test_get_key_dimensions():
    """Test extracting key dimensions from hidden state."""
    # Arrange
    hidden_state = np.array([0.1, 0.5, -0.8, 0.2, -0.3])
    state = LatentState(hidden_state=hidden_state)
    
    # Act
    key_dims = state.get_key_dimensions(top_k=3)
    
    # Assert
    assert len(key_dims) == 3
    # Indices of highest magnitude values should be returned (2, 1, 4)
    assert 2 in key_dims  # -0.8
    assert 1 in key_dims  # 0.5
    assert 4 in key_dims  # -0.3


def test_to_dict(sample_state):
    """Test converting state to dictionary representation."""
    # Act
    state_dict = sample_state.to_dict()
    
    # Assert
    assert isinstance(state_dict, dict)
    assert "iteration" in state_dict
    assert state_dict["iteration"] == sample_state.iteration
    assert "key_concepts" in state_dict
    assert isinstance(state_dict["key_concepts"], list)
    assert "attention_weights" in state_dict
    assert "reasoning_path" in state_dict
    assert "hidden_state_norm" in state_dict
    assert "key_dimensions" in state_dict
