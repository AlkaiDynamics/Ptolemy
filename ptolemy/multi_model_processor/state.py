from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import copy
import uuid

@dataclass
class ProcessorState:
    """
    Tracks the state of a task through the multi-model processing pipeline.
    This class provides transparency into the processing stages, model selection,
    and performance metrics throughout the lifecycle of a task.
    """
    
    # Core state elements
    task_id: str
    task: str
    context: Optional[Dict[str, Any]] = None
    
    # Processing stage tracking
    current_stage: str = "initialized"
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model selection info
    selected_model: Optional[str] = None
    model_candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Reasoning integration
    reasoning_output: Optional[Dict[str, Any]] = None
    reasoning_iterations: int = 0
    
    # Performance tracking
    start_time: float = field(default_factory=time.time)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    
    # Final output
    response: Optional[Dict[str, Any]] = None
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def create(cls, task: str, context: Optional[Dict[str, Any]] = None) -> 'ProcessorState':
        """
        Factory method to create a new ProcessorState with a generated UUID.
        
        Args:
            task: The task description or query
            context: Optional context information
            
        Returns:
            A new ProcessorState instance
        """
        return cls(
            task_id=str(uuid.uuid4()),
            task=task,
            context=context
        )
    
    def advance_stage(self, new_stage: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Advance to a new processing stage and record timing information.
        
        Args:
            new_stage: Name of the new stage
            metadata: Optional metadata about the stage transition
        """
        now = time.time()
        
        # Record timing for the previous stage
        if self.current_stage != "initialized":
            elapsed = now - self.stage_timings.get(f"{self.current_stage}_start", self.start_time)
            self.stage_timings[self.current_stage] = elapsed
        
        # Record new stage
        self.stage_timings[f"{new_stage}_start"] = now
        self.stage_history.append({
            "stage": new_stage,
            "previous_stage": self.current_stage,
            "timestamp": now,
            "metadata": metadata or {}
        })
        self.current_stage = new_stage
    
    def record_token_usage(self, stage: str, input_tokens: int, output_tokens: int) -> None:
        """
        Record token usage for a processing stage.
        
        Args:
            stage: Name of the processing stage
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
        """
        self.token_usage[f"{stage}_input"] = input_tokens
        self.token_usage[f"{stage}_output"] = output_tokens
        
        # Update totals
        self.token_usage["total_input"] = sum(
            v for k, v in self.token_usage.items() if k.endswith("_input")
        )
        self.token_usage["total_output"] = sum(
            v for k, v in self.token_usage.items() if k.endswith("_output")
        )
    
    def record_model_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Record model candidates considered for this task.
        
        Args:
            candidates: List of model candidates with scores and capabilities
        """
        self.model_candidates = candidates
    
    def select_model(self, model_id: str, reason: Optional[str] = None) -> None:
        """
        Record the selected model for this task.
        
        Args:
            model_id: Identifier of the selected model
            reason: Optional reason for selection
        """
        self.selected_model = model_id
        self.stage_history.append({
            "stage": "model_selection",
            "previous_stage": self.current_stage,
            "timestamp": time.time(),
            "metadata": {"model_id": model_id, "reason": reason}
        })
    
    def record_error(self, error: Exception, stage: str, severity: str = "error") -> None:
        """
        Record an error that occurred during processing.
        
        Args:
            error: The exception that occurred
            stage: Stage where the error occurred
            severity: Error severity level
        """
        self.errors.append({
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
            "severity": severity,
            "timestamp": time.time()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the state
        """
        return {
            "task_id": self.task_id,
            "stages": [stage["stage"] for stage in self.stage_history],
            "current_stage": self.current_stage,
            "selected_model": self.selected_model,
            "reasoning_iterations": self.reasoning_iterations,
            "total_time": time.time() - self.start_time,
            "stage_timings": self.stage_timings,
            "token_usage": self.token_usage,
            "errors": self.errors
        }
    
    def get_stage_timing(self, stage: str) -> Optional[float]:
        """
        Get the time spent in a specific stage.
        
        Args:
            stage: Name of the stage
            
        Returns:
            Time in seconds or None if stage not completed
        """
        return self.stage_timings.get(stage)
    
    def get_total_time(self) -> float:
        """
        Get the total processing time so far.
        
        Returns:
            Total time in seconds
        """
        return time.time() - self.start_time