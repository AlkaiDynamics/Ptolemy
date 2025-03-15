# ptolemy/latent_reasoning/metrics.py
from typing import Dict, List, Any, Optional
import time
import json
from collections import deque
from pathlib import Path
import statistics
from loguru import logger

class ReasoningMetrics:
    """
    Tracks and analyzes metrics for the latent reasoning process
    to enable optimization and debugging.
    """
    
    def __init__(self, max_history: int = 1000):
        self.process_history = deque(maxlen=max_history)
        self.task_type_stats = {}
        self.total_processes = 0
        self.total_iterations = 0
        self.total_time = 0.0
        
        # Timing metrics
        self.start_time = time.time()
        
        logger.info(f"Initialized ReasoningMetrics with history size {max_history}")
    
    def record_process(
        self, 
        process_id: str, 
        task_type: str, 
        iterations: int, 
        process_time: float,
        convergence: float
    ) -> None:
        """
        Record metrics for a reasoning process.
        
        Args:
            process_id: Unique identifier for the process
            task_type: Type of task being processed
            iterations: Number of iterations performed
            process_time: Time taken in seconds
            convergence: Final convergence value
        """
        # Create process record
        process = {
            "id": process_id,
            "task_type": task_type,
            "iterations": iterations,
            "time": process_time,
            "convergence": convergence,
            "timestamp": time.time()
        }
        
        # Add to history
        self.process_history.append(process)
        
        # Update aggregate metrics
        self.total_processes += 1
        self.total_iterations += iterations
        self.total_time += process_time
        
        # Update task type stats
        if task_type not in self.task_type_stats:
            self.task_type_stats[task_type] = {
                "count": 0,
                "total_iterations": 0,
                "total_time": 0.0,
                "iteration_history": []
            }
        
        self.task_type_stats[task_type]["count"] += 1
        self.task_type_stats[task_type]["total_iterations"] += iterations
        self.task_type_stats[task_type]["total_time"] += process_time
        self.task_type_stats[task_type]["iteration_history"].append(iterations)
        
        # Keep iteration history manageable
        if len(self.task_type_stats[task_type]["iteration_history"]) > 100:
            self.task_type_stats[task_type]["iteration_history"] = self.task_type_stats[task_type]["iteration_history"][-100:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of reasoning metrics."""
        uptime = time.time() - self.start_time
        
        summary = {
            "total_processes": self.total_processes,
            "total_iterations": self.total_iterations,
            "total_processing_time": self.total_time,
            "system_uptime": uptime,
            "avg_iterations_per_process": self.total_iterations / max(1, self.total_processes),
            "avg_time_per_process": self.total_time / max(1, self.total_processes),
            "avg_time_per_iteration": self.total_time / max(1, self.total_iterations),
            "task_types": {}
        }
        
        # Add task type specific stats
        for task_type, stats in self.task_type_stats.items():
            iterations_history = stats["iteration_history"]
            
            task_summary = {
                "count": stats["count"],
                "avg_iterations": stats["total_iterations"] / max(1, stats["count"]),
                "avg_time": stats["total_time"] / max(1, stats["count"]),
                "median_iterations": statistics.median(iterations_history) if iterations_history else 0,
                "min_iterations": min(iterations_history) if iterations_history else 0,
                "max_iterations": max(iterations_history) if iterations_history else 0
            }
            
            summary["task_types"][task_type] = task_summary
        
        return summary
    
    def save_metrics(self, filepath: str) -> bool:
        """
        Save metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metrics = {
                "summary": self.get_summary(),
                "history": list(self.process_history)[-100:]  # Save last 100 processes
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved reasoning metrics to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            return False
