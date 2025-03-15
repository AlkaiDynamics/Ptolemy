# ptolemy/latent_reasoning/state.py
from typing import Dict, List, Any, Optional, Set
import numpy as np
import copy
from dataclasses import dataclass, field

@dataclass
class LatentState:
    """
    Represents the latent reasoning state that is processed iteratively.
    This state carries information through the recurrent iterations.
    """
    
    # Core state elements
    hidden_state: np.ndarray  # The main hidden state vector (high-dimensional)
    context_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)  # Embeddings of context elements
    task_embedding: Optional[np.ndarray] = None  # Embedding of the task
    
    # State metadata
    iteration: int = 0  # Current iteration number
    key_concepts: Set[str] = field(default_factory=set)  # Key concepts being reasoned about
    attention_weights: Dict[str, float] = field(default_factory=dict)  # Attention to different context elements
    reasoning_path: List[str] = field(default_factory=list)  # Sequence of reasoning steps
    
    # Convergence tracking
    prev_norms: List[float] = field(default_factory=list)  # History of state norms for convergence tracking
    
    def copy_state(self) -> 'LatentState':
        """Create a deep copy of the state."""
        return copy.deepcopy(self)
    
    def update_hidden_state(self, new_hidden_state: np.ndarray) -> None:
        """Update the hidden state."""
        self.hidden_state = new_hidden_state
        self.prev_norms.append(np.linalg.norm(new_hidden_state))
    
    def add_key_concept(self, concept: str) -> None:
        """Add a key concept being reasoned about."""
        self.key_concepts.add(concept)
    
    def update_attention(self, context_key: str, weight: float) -> None:
        """Update attention weight for a context element."""
        self.attention_weights[context_key] = weight
    
    def add_reasoning_step(self, step_description: str) -> None:
        """Add a reasoning step to the path."""
        self.reasoning_path.append(step_description)
    
    def calculate_change(self, previous_state: 'LatentState') -> float:
        """
        Calculate the relative change between this state and a previous state.
        Used for convergence detection.
        """
        if previous_state is None:
            return 1.0
            
        # Calculate cosine similarity between hidden states
        dot_product = np.dot(self.hidden_state, previous_state.hidden_state)
        norm_current = np.linalg.norm(self.hidden_state)
        norm_previous = np.linalg.norm(previous_state.hidden_state)
        
        if norm_current == 0 or norm_previous == 0:
            return 1.0
            
        similarity = dot_product / (norm_current * norm_previous)
        
        # Convert similarity to distance/change
        return 1.0 - similarity
    
    def get_key_dimensions(self, top_k: int = 10) -> List[int]:
        """Get the indices of the key dimensions in the hidden state with highest values."""
        if self.hidden_state is None or len(self.hidden_state) == 0:
            return []
            
        # Find indices of top-k values by magnitude
        indices = np.argsort(np.abs(self.hidden_state))[-top_k:]
        return indices.tolist()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "key_concepts": list(self.key_concepts),
            "attention_weights": self.attention_weights,
            "reasoning_path": self.reasoning_path,
            "hidden_state_norm": float(np.linalg.norm(self.hidden_state)) if self.hidden_state is not None else 0.0,
            "key_dimensions": self.get_key_dimensions()
        }
