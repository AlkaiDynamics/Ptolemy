"""
State management for the Multi-Model Processor.

This module defines the state tracking objects used by the 
Multi-Model Processor to maintain context between processing
stages and track performance metrics.
"""

import uuid
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from loguru import logger

class ProcessorState:
    """
    Tracks the state of a task through the processing pipeline.
    
    This class maintains the core state elements for a task
    as it passes through the processor, as well as collecting
    performance metrics and handling error conditions.
    """
    
    def __init__(
        self, 
        task_id: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new processor state.
        
        Args:
            task_id: Unique identifier for the task
            task: The task text
            context: Optional context dictionary
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")
            
        if not task_id or not isinstance(task_id, str):
            raise ValueError("Task ID must be a non-empty string")
            
        # Core task information
        self.task_id = task_id
        self.task = task
        self.context = context or {}
        
        # Stage tracking
        self.current_stage = "created"
        self.stage_history = [("created", time.time())]
        
        # Processing metadata
        self.start_time = time.time()
        self.end_time = None
        self.processing_time = None
        self.expiry_time = None
        
        # Result data
        self.output = None
        self.error = None
        self.model_used = None
        self.cached = False
        
        # Performance metrics
        self.metrics = {
            "reasoning_time": None,
            "model_selection_time": None,
            "processing_time": None,
            "tokens_input": 0,
            "tokens_output": 0
        }
        
        # Reasoning tracking
        self.reasoning = []
        self.reasoning_iterations = 0
        
        # Streaming state
        self.streaming = False
        self.stream_chunks = []
        
        logger.debug(f"Created new processor state for task {task_id}")
    
    @classmethod
    def create(cls, task: str, context: Optional[Dict[str, Any]] = None) -> 'ProcessorState':
        """
        Create a new processor state with a generated UUID.
        
        Args:
            task: The task to process
            context: Optional context for the task
            
        Returns:
            New processor state instance
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")
            
        task_id = str(uuid.uuid4())
        return cls(
            task_id=task_id,
            task=task,
            context=context
        )
    
    def set_stage(self, stage: str) -> None:
        """
        Update the current processing stage.
        
        Args:
            stage: New stage name
        """
        self.current_stage = stage
        self.stage_history.append((stage, time.time()))
        logger.debug(f"Task {self.task_id} moved to stage: {stage}")
    
    def add_reasoning(self, step: str) -> None:
        """
        Add a reasoning step.
        
        Args:
            step: Reasoning step text
        """
        self.reasoning.append(step)
        self.reasoning_iterations += 1
    
    def set_error(self, error: str) -> None:
        """
        Set error information.
        
        Args:
            error: Error message
        """
        self.error = error
        self.set_stage("error")
        logger.error(f"Task {self.task_id} error: {error}")
    
    def set_model(self, model: str) -> None:
        """
        Set the model used for processing.
        
        Args:
            model: Model identifier
        """
        self.model_used = model
        self.set_stage("model_selected")
    
    def set_output(self, output: str) -> None:
        """
        Set the task output.
        
        Args:
            output: Task output text
        """
        self.output = output
        self.set_stage("completed")
    
    def set_cached(self, cached: bool) -> None:
        """
        Set the cached flag.
        
        Args:
            cached: Whether the result was from cache
        """
        self.cached = cached
        
    def add_stream_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Add a streaming output chunk.
        
        Args:
            chunk: Stream chunk data
        """
        self.stream_chunks.append(chunk)
        self.streaming = True
    
    def update_metrics(self, metric_name: str, value: Any) -> None:
        """
        Update a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name in self.metrics:
            self.metrics[metric_name] = value
    
    def finalize(self) -> None:
        """
        Finalize the state when processing is complete.
        This updates timing metrics and can be used for cleanup.
        """
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.metrics["processing_time"] = self.processing_time
        
        # Calculate stage times
        stages = {}
        for i in range(1, len(self.stage_history)):
            stage_name, stage_start = self.stage_history[i-1]
            _, stage_end = self.stage_history[i]
            duration = stage_end - stage_start
            stages[stage_name] = duration
            
        self.metrics["stage_times"] = stages
        logger.debug(f"Task {self.task_id} finalized. Processing time: {self.processing_time:.2f}s")
    
    def set_expiry(self, expiry_time: float) -> None:
        """
        Set an expiration time for this state.
        
        Args:
            expiry_time: Unix timestamp when this state should expire
        """
        self.expiry_time = expiry_time
        
    def is_expired(self) -> bool:
        """
        Check if this state has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expiry_time is None:
            return False
        return time.time() > self.expiry_time
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.
        
        Returns:
            Dictionary representation of state
        """
        return {
            "task_id": self.task_id,
            "task": self.task,
            "current_stage": self.current_stage,
            "start_time": self.start_time,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "cached": self.cached,
            "error": self.error,
            "metrics": self.metrics,
            "reasoning_iterations": self.reasoning_iterations
        }