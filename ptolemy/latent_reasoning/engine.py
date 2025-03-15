# ptolemy/latent_reasoning/engine.py
from typing import Dict, List, Optional, Any, Union
import asyncio
import numpy as np
import uuid
from datetime import datetime
from loguru import logger
import time

from ptolemy.context_engine.core import ContextEngine
from .state import LatentState
from .recurrent_block import RecurrentBlock
from .metrics import ReasoningMetrics
from .adapters.context_adapter import ContextAdapter
from .adapters.model_adapter import ModelAdapter

class LatentReasoningEngine:
    """
    Processes context and task information through recurrent iterations to produce 
    enhanced reasoning before sending to language models.
    """
    
    def __init__(
        self, 
        context_engine: ContextEngine,
        hidden_dim: int = 512,
        default_iterations: int = 8,
        max_iterations: int = 32,
        convergence_threshold: float = 0.05
    ):
        """
        Initialize the Latent Reasoning Engine.
        
        Args:
            context_engine: Enhanced Context Engine instance
            hidden_dim: Dimension of hidden state
            default_iterations: Default number of iterations
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence detection
        """
        self.context_engine = context_engine
        self.hidden_dim = hidden_dim
        self.default_iterations = default_iterations
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize components
        self.recurrent_block = RecurrentBlock(hidden_dim=hidden_dim)
        self.metrics = ReasoningMetrics()
        self.context_adapter = ContextAdapter(hidden_dim=hidden_dim)
        self.model_adapter = ModelAdapter()
        
        logger.info("Latent Reasoning Engine initialized")
    
    async def process(
        self, 
        task: str, 
        relevant_components: Optional[List[str]] = None,
        iterations: Optional[int] = None,
        adaptive: bool = True,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task through latent space reasoning.
        
        Args:
            task: The task description to process
            relevant_components: Optional list of relevant component IDs
            iterations: Number of reasoning iterations (uses default if None)
            adaptive: Whether to use adaptive iteration stopping
            task_metadata: Additional metadata about the task
            
        Returns:
            Dictionary containing processed output and reasoning metadata
        """
        start_time = time.time()
        process_id = str(uuid.uuid4())
        
        try:
            # Set iterations
            if iterations is None:
                iterations = self.default_iterations
            iterations = min(iterations, self.max_iterations)
            
            logger.info(f"Starting latent reasoning process {process_id} with max {iterations} iterations")
            
            # Get rich context from the enhanced Context Engine
            context = await self.context_engine.get_model_context(task, relevant_components)
            
            # Prepare the initial state
            initial_state = await self.context_adapter.prepare_initial_state(task, context, task_metadata)
            
            # Initialize state history for tracking
            state_history = [initial_state.copy_state()]
            
            # Run the recurrent process
            current_state = initial_state
            actual_iterations = 0
            adaptive_stopped = False
            
            for i in range(iterations):
                previous_state = current_state.copy_state()
                
                # Apply one step of recurrent processing
                current_state = await self.recurrent_block.process_step(
                    current_state, 
                    task=task, 
                    context=context,
                    step=i
                )
                
                # Store state in history
                state_history.append(current_state.copy_state())
                actual_iterations = i + 1
                
                # Check for convergence if using adaptive stopping
                if adaptive and i > 0:
                    convergence = current_state.calculate_change(previous_state)
                    logger.debug(f"Iteration {i+1}: change = {convergence}")
                    
                    if convergence < self.convergence_threshold:
                        logger.info(f"Adaptive stopping triggered after {i+1} iterations (change: {convergence} < threshold: {self.convergence_threshold})")
                        adaptive_stopped = True
                        actual_iterations = i + 1
                        break
                
            # Prepare final output for the model
            output = await self.model_adapter.prepare_output(current_state, task, context)
            
            # Record metrics
            end_time = time.time()
            process_time = end_time - start_time
            
            self.metrics.record_process(
                process_id=process_id,
                task_type=task_metadata.get("type", "unknown") if task_metadata else "unknown",
                iterations=actual_iterations,
                process_time=process_time,
                convergence=current_state.calculate_change(state_history[-2]) if len(state_history) > 1 else 1.0
            )
            
            # Return processed result with metadata
            return {
                "process_id": process_id,
                "output": output,
                "iterations": actual_iterations,
                "time": process_time,
                "adaptive_stopped": adaptive_stopped,
                "state_trajectory": self._extract_trajectory_summary(state_history) if task_metadata and task_metadata.get("include_trajectory", False) else None
            }
            
        except Exception as e:
            logger.error(f"Error in latent reasoning process: {str(e)}")
            
            # Calculate elapsed time
            end_time = time.time()
            process_time = end_time - start_time
            
            # Return minimal result with error info
            return {
                "process_id": process_id,
                "error": str(e),
                "output": {
                    "context": context if 'context' in locals() else "Error retrieving context",
                    "key_concepts": [],
                    "attention_focus": {},
                    "reasoning": f"Error during latent reasoning: {str(e)}"
                },
                "iterations": 0,
                "time": process_time,
                "adaptive_stopped": False,
                "state_trajectory": None
            }
    
    def _extract_trajectory_summary(self, state_history: List[LatentState]) -> Dict[str, Any]:
        """Extract a summary of the state trajectory for analysis."""
        if not state_history:
            return {}
            
        # Create a simplified representation of the trajectory
        trajectory = {
            "steps": len(state_history),
            "convergence_pattern": [
                state_history[i].calculate_change(state_history[i-1]) 
                for i in range(1, len(state_history))
            ],
            "key_dimensions": state_history[-1].get_key_dimensions()
        }
        
        return trajectory
    
    async def analyze_task_complexity(self, task: str, context: str) -> Dict[str, Any]:
        """
        Analyze task complexity to determine optimal iteration count.
        
        Args:
            task: The task description
            context: The context for the task
            
        Returns:
            Dictionary with complexity analysis
        """
        # Simple heuristics for task complexity
        complexity_indicators = {
            "length": len(task),
            "context_size": len(context),
            "questions": task.count("?"),
            "code_blocks": task.count("```"),
            "steps_required": task.lower().count("step") + task.lower().count("first") + task.lower().count("then")
        }
        
        # Calculate a complexity score
        base_score = 10.0
        length_factor = min(complexity_indicators["length"] / 100, 5.0)
        context_factor = min(complexity_indicators["context_size"] / 1000, 5.0)
        question_factor = complexity_indicators["questions"] * 2.0
        code_factor = complexity_indicators["code_blocks"] * 3.0
        steps_factor = complexity_indicators["steps_required"] * 1.5
        
        complexity_score = base_score + length_factor + context_factor + question_factor + code_factor + steps_factor
        
        # Map to recommended iterations
        recommended_iterations = max(4, min(self.max_iterations, int(complexity_score)))
        
        return {
            "complexity_score": complexity_score,
            "recommended_iterations": recommended_iterations,
            "indicators": complexity_indicators
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of reasoning metrics."""
        return self.metrics.get_summary()
