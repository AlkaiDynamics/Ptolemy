# ptolemy/latent_reasoning/adapters/model_adapter.py
from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger

from ..state import LatentState

class ModelAdapter:
    """
    Prepares the output of latent reasoning for consumption by language models.
    """
    
    def __init__(self):
        logger.info("Initialized ModelAdapter")
    
    async def prepare_output(
        self, 
        state: LatentState, 
        task: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Prepare the final output from the latent reasoning process.
        
        Args:
            state: Final latent state after reasoning
            task: Original task description
            context: Original context information
            
        Returns:
            Dictionary with processed outputs for model consumption
        """
        try:
            # Extract key concepts from final state
            key_concepts = list(state.key_concepts)
            
            # Extract attention focus from final state
            attention_focus = self._get_attention_focus(state.attention_weights)
            
            # Create reasoning summary
            reasoning_summary = self._create_reasoning_summary(state.reasoning_path)
            
            # Assemble enhanced prompt
            enhanced_context = self._create_enhanced_context(
                task=task,
                original_context=context,
                key_concepts=key_concepts,
                attention_focus=attention_focus,
                reasoning_summary=reasoning_summary
            )
            
            # Create final output object
            output = {
                "context": enhanced_context,
                "key_concepts": key_concepts,
                "attention_focus": attention_focus,
                "reasoning": reasoning_summary,
                "iterations": state.iteration
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error preparing model output: {str(e)}")
            
            # Return minimal output in case of error
            return {
                "context": context,
                "reasoning": f"Error preparing model output: {str(e)}",
                "iterations": state.iteration if hasattr(state, 'iteration') else 0
            }
    
    def _get_attention_focus(self, attention_weights: Dict[str, float]) -> Dict[str, float]:
        """Extract the top focus areas based on attention weights."""
        if not attention_weights:
            return {}
            
        # Sort by weight, descending
        sorted_weights = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 5 focus areas
        top_focus = {k: v for k, v in sorted_weights[:5]}
        
        return top_focus
    
    def _create_reasoning_summary(self, reasoning_path: List[str]) -> str:
        """Create a summary of the reasoning process."""
        if not reasoning_path:
            return "No reasoning steps available."
            
        # For brevity, only include key steps
        if len(reasoning_path) > 10:
            # Include first step, last step, and a few from the middle
            key_steps = [
                reasoning_path[0],
                *reasoning_path[1:4],
                "...",
                *reasoning_path[-3:]
            ]
            summary = "\n".join(key_steps)
        else:
            summary = "\n".join(reasoning_path)
            
        return summary
    
    def _create_enhanced_context(
        self,
        task: str,
        original_context: str,
        key_concepts: List[str],
        attention_focus: Dict[str, float],
        reasoning_summary: str
    ) -> str:
        """
        Create an enhanced context for the model based on reasoning results.
        
        Args:
            task: Original task description
            original_context: Original context information
            key_concepts: Key concepts identified during reasoning
            attention_focus: Areas of focus based on attention
            reasoning_summary: Summary of the reasoning process
            
        Returns:
            Enhanced context string
        """
        # Assemble enhanced context
        sections = []
        
        # Add header with key concepts
        sections.append("ENHANCED CONTEXT (with latent reasoning)")
        sections.append(f"Key concepts: {', '.join(key_concepts)}")
        
        # Add task with reasoning insights
        sections.append(f"\nTASK: {task}")
        
        # Add attention focus areas
        if attention_focus:
            focus_sections = []
            for area, weight in attention_focus.items():
                focus_sections.append(f"- {area}: {weight:.2f}")
            sections.append("\nFOCUS AREAS:\n" + "\n".join(focus_sections))
        
        # Add original context
        sections.append(f"\nCONTEXT:\n{original_context}")
        
        # Add reasoning trace (optional, could be omitted in production)
        # sections.append(f"\nREASONING TRACE:\n{reasoning_summary}")
        
        # Join all sections
        enhanced_context = "\n".join(sections)
        
        return enhanced_context
