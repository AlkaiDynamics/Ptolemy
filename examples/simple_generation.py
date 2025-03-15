#!/usr/bin/env python
"""
Simple example demonstrating how to use the PTOLEMY system for code generation.
This example uses mock responses when the OpenAI API is not available.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the ptolemy package
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from ptolemy.temporal_core import TemporalCore
from ptolemy.context_engine import ContextEngine
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.feedback import FeedbackOrchestrator


async def main():
    """Run a simple example of using PTOLEMY for code generation."""
    # Initialize components
    temporal_core = TemporalCore()
    context_engine = ContextEngine(temporal_core)
    
    try:
        multi_model = MultiModelProcessor(context_engine)
    except Exception as e:
        logger.error(f"Error initializing MultiModelProcessor: {str(e)}")
        logger.info("Using mock implementation for demonstration purposes")
        
        # Create a mock MultiModelProcessor
        class MockMultiModelProcessor:
            async def route_task(self, task, model_type, options=None):
                return f"[MOCK RESPONSE] This is a simulated response for: {task}"
                
            async def route_multi_stage(self, task, stages, options=None):
                results = []
                for stage in stages:
                    results.append({
                        "stage": stage["name"],
                        "output": f"[MOCK RESPONSE] This is a simulated response for stage {stage['name']}: {task}"
                    })
                return results
        
        multi_model = MockMultiModelProcessor()
    
    feedback_orchestrator = FeedbackOrchestrator(temporal_core, context_engine)
    
    # Initialize the system
    await temporal_core.initialize()
    await context_engine.initialize()
    
    print("PTOLEMY system initialized successfully")
    
    # Store some example patterns and insights
    await context_engine.store_pattern(
        "error_handling",
        "python",
        """
def safe_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper
        """,
        {"language": "python", "category": "error_handling"}
    )
    
    await context_engine.store_insight(
        "code_style",
        "Use descriptive variable names and add docstrings to all functions",
        0.9,
        {"source": "user_preference"}
    )
    
    # Generate code using the implementer model
    prompt = "Create a Python function to calculate the Fibonacci sequence up to n terms"
    print(f"\nGenerating code for: {prompt}")
    
    result = await multi_model.route_task(prompt, "implementer")
    
    print("\nGenerated code:")
    print("===============")
    print(result)
    
    # Record feedback on the generated code
    feedback = "The code is good, but I would prefer more detailed comments"
    await feedback_orchestrator.record_feedback(
        "code_quality",
        feedback,
        "user",
        None,
        {"sentiment": "neutral", "model_type": "implementer"}
    )
    
    # Generate improved code with the feedback incorporated
    prompt_with_feedback = f"{prompt}\n\nAdditional requirements: {feedback}"
    print(f"\nGenerating improved code with feedback: {feedback}")
    
    improved_result = await multi_model.route_task(prompt_with_feedback, "implementer")
    
    print("\nImproved code:")
    print("==============")
    print(improved_result)
    
    # Run a multi-stage chain
    print("\nRunning a multi-stage prompt chain...")
    stages = [
        {"name": "architecture", "model_type": "architect", "next_prompt": "Implement this architecture"},
        {"name": "implementation", "model_type": "implementer", "next_prompt": "Review this implementation"},
        {"name": "review", "model_type": "reviewer"}
    ]
    
    chain_results = await multi_model.route_multi_stage(
        "Design a simple API for a todo list application",
        stages
    )
    
    print("\nChain Results:")
    print("==============")
    for result in chain_results:
        print(f"\n## Stage: {result['stage']}")
        print(result["output"])


if __name__ == "__main__":
    asyncio.run(main())
