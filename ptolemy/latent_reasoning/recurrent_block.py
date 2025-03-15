# ptolemy/latent_reasoning/recurrent_block.py
from typing import Dict, List, Any, Optional
import numpy as np
import asyncio
from loguru import logger

from .state import LatentState

class RecurrentBlock:
    """
    Implements the core recurrent processing block that iteratively 
    updates the latent state based on context and task.
    """
    
    def __init__(self, hidden_dim: int = 512):
        self.hidden_dim = hidden_dim
        
        # Initialize weights for the recurrent operation
        # In a more sophisticated implementation, these could be learned weights
        np.random.seed(42)  # For reproducibility
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim) * 0.01  # Hidden state transformation
        self.W_context = np.random.randn(hidden_dim, hidden_dim) * 0.01  # Context integration
        self.W_task = np.random.randn(hidden_dim, hidden_dim) * 0.01     # Task integration
        
        # Initialize layer normalization parameters
        self.gamma = np.ones(hidden_dim)
        self.beta = np.zeros(hidden_dim)
        
        logger.info(f"Initialized RecurrentBlock with hidden dimension {hidden_dim}")
    
    async def process_step(
        self, 
        state: LatentState, 
        task: str, 
        context: str,
        step: int
    ) -> LatentState:
        """
        Process one step of the recurrent reasoning.
        
        Args:
            state: Current latent state
            task: The task description
            context: The context information
            step: Current iteration step
            
        Returns:
            Updated latent state
        """
        # Create a new state to avoid modifying the input state
        new_state = state.copy_state()
        new_state.iteration = step + 1
        
        try:
            # Apply recurrent transformation
            hidden_transform = np.dot(new_state.hidden_state, self.W_hidden)
            
            # Apply context integration (simplified)
            context_vector = np.zeros(self.hidden_dim)
            for key, embedding in new_state.context_embeddings.items():
                # Calculate attention weight (simplified)
                attention = self._calculate_attention(new_state.hidden_state, embedding)
                new_state.update_attention(key, float(attention))
                
                # Integrate weighted context
                context_vector += attention * np.dot(embedding, self.W_context)
            
            # Apply task integration
            task_vector = np.zeros(self.hidden_dim)
            if new_state.task_embedding is not None:
                task_vector = np.dot(new_state.task_embedding, self.W_task)
            
            # Combine transformations
            combined = hidden_transform + context_vector + task_vector
            
            # Apply non-linearity (ReLU)
            activated = np.maximum(0, combined)
            
            # Apply layer normalization for stability
            normalized = self._layer_normalize(activated)
            
            # Update the hidden state
            new_state.update_hidden_state(normalized)
            
            # Analyze the state to extract key concepts (simplified)
            self._extract_key_concepts(new_state, task, context)
            
            # Add reasoning step description
            new_state.add_reasoning_step(f"Iteration {step+1}: Processed task with context integration")
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error in recurrent processing step {step}: {str(e)}")
            # Return the original state in case of error
            state.add_reasoning_step(f"Error in iteration {step+1}: {str(e)}")
            return state
    
    def _calculate_attention(self, hidden_state: np.ndarray, embedding: np.ndarray) -> float:
        """Calculate attention weight between hidden state and an embedding."""
        # Simple dot product attention with softmax normalization
        dot_product = np.dot(hidden_state, embedding)
        
        # Apply softmax-like normalization
        return 1.0 / (1.0 + np.exp(-dot_product))  # Sigmoid function
    
    def _layer_normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization for stability."""
        # Calculate mean and variance
        mean = np.mean(x)
        variance = np.var(x)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + 1e-5)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta
    
    def _extract_key_concepts(self, state: LatentState, task: str, context: str) -> None:
        """Extract key concepts being reasoned about (simplified)."""
        # This is a simplified placeholder implementation
        # In a real system, this would involve more sophisticated analysis
        
        # Simple keyword extraction
        key_terms = []
        
        # Extract from task
        words = task.lower().split()
        for word in words:
            if len(word) > 4 and word not in ["should", "would", "could", "with", "that", "this", "these", "those"]:
                key_terms.append(word)
        
        # Extract a few context keywords
        context_sample = " ".join(context.split()[:100])  # Sample first 100 words
        context_words = context_sample.lower().split()
        for word in context_words:
            if len(word) > 5 and word not in key_terms:
                key_terms.append(word)
                if len(key_terms) >= 10:
                    break
        
        # Add to state
        for term in key_terms[:5]:  # Limit to top 5
            state.add_key_concept(term)
