# ptolemy/latent_reasoning/adapters/context_adapter.py
from typing import Dict, List, Any, Optional
import numpy as np
import re
from loguru import logger

from ..state import LatentState

class ContextAdapter:
    """
    Prepares context and task information for latent reasoning by
    creating appropriate representations for the reasoning process.
    """
    
    def __init__(self, hidden_dim: int = 512):
        """
        Initialize the ContextAdapter.
        
        Args:
            hidden_dim: Dimension of the hidden state
        """
        self.hidden_dim = hidden_dim
        logger.info(f"Initialized ContextAdapter with hidden dimension {hidden_dim}")
    
    async def prepare_initial_state(
        self, 
        task: str, 
        context: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> LatentState:
        """
        Prepare initial latent state from task and context.
        
        Args:
            task: The task description
            context: The context information
            metadata: Optional additional metadata
            
        Returns:
            Initial latent state
        """
        try:
            # Create task embedding (simplified)
            task_embedding = await self._create_embedding(task)
            
            # Parse context into meaningful segments
            context_segments = self._parse_context(context)
            
            # Create embeddings for context segments
            context_embeddings = {}
            for key, text in context_segments.items():
                embedding = await self._create_embedding(text)
                context_embeddings[key] = embedding
            
            # Initialize random hidden state
            np.random.seed(42)  # For reproducibility
            hidden_state = np.random.randn(self.hidden_dim) * 0.01
            
            # Create initial state
            initial_state = LatentState(
                hidden_state=hidden_state,
                context_embeddings=context_embeddings,
                task_embedding=task_embedding,
                iteration=0
            )
            
            # Extract initial key concepts
            key_concepts = self._extract_initial_concepts(task, context)
            for concept in key_concepts:
                initial_state.add_key_concept(concept)
            
            # Initialize reasoning path
            initial_state.add_reasoning_step("Initialization: Created initial state from task and context")
            
            return initial_state
            
        except Exception as e:
            logger.error(f"Error preparing initial state: {str(e)}")
            # Return a minimal valid state in case of error
            np.random.seed(42)
            hidden_state = np.random.randn(self.hidden_dim) * 0.01
            
            error_state = LatentState(
                hidden_state=hidden_state,
                context_embeddings={},
                task_embedding=np.zeros(self.hidden_dim),
                iteration=0
            )
            
            error_state.add_reasoning_step(f"Error during initialization: {str(e)}")
            return error_state
    
    async def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create an embedding for text.
        
        This is a simplified mock implementation. In a real system,
        this would use a proper embedding model.
        """
        # Create a deterministic "embedding" based on text hash
        import hashlib
        
        # Create a deterministic hash of the text
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        # Use only the first 8 characters of the hash and make sure it's within the valid range
        hash_int = int(hash_hex[:8], 16) % (2**32 - 1)
        
        # Use the hash to seed a random number generator
        np.random.seed(hash_int)
        
        # Generate a random embedding vector
        embedding = np.random.rand(self.hidden_dim) - 0.5  # Center around 0
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _parse_context(self, context: str) -> Dict[str, str]:
        """
        Parse context string into meaningful segments.
        
        Args:
            context: The context string to parse
            
        Returns:
            Dictionary of context segments
        """
        if not context:
            return {}
            
        # Simple section-based parsing
        sections = {}
        
        # Try to split by section headers
        section_pattern = r'([A-Z\s]+):\s*\n'
        matches = re.finditer(section_pattern, context)
        
        last_pos = 0
        current_section = "general"
        
        for match in matches:
            # Add text before this section header to the current section
            section_text = context[last_pos:match.start()].strip()
            if section_text and last_pos > 0:  # Skip empty sections and the text before the first header
                sections[current_section] = section_text
            
            # Update current section and position
            current_section = match.group(1).strip()
            last_pos = match.end()
        
        # Add the last section
        if last_pos < len(context):
            sections[current_section] = context[last_pos:].strip()
        
        # If no sections were found, use the whole context
        if not sections:
            sections["general"] = context
            
        return sections
    
    def _extract_initial_concepts(self, task: str, context: str) -> List[str]:
        """
        Extract initial key concepts from task and context.
        
        Args:
            task: Task description
            context: Context information
            
        Returns:
            List of key concepts
        """
        # Simple keyword extraction (placeholder implementation)
        words = (task + " " + context[:500]).lower().split()
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "with", "by", "to", "for", 
                     "and", "or", "but", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "should", "could", "this", "that", "these", "those"}
        
        # Count word frequency
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 10 concepts
        return [word for word, _ in sorted_words[:10]]
