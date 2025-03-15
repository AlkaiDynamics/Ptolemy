#!/usr/bin/env python
"""
Multi-Model Processor Demo Script

This script demonstrates the functionality of the Multi-Model Processor 
by processing sample tasks and showing the results.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
import time
from typing import Dict, Any, Optional

# Add the parent directory to sys.path to allow importing the ptolemy package
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ptolemy.multi_model_processor.processor import ProcessorEngine
from ptolemy.multi_model_processor.state import ProcessorState
from ptolemy.multi_model_processor.handlers.mock_handler import MockModelHandler


async def run_simple_demo():
    """Run a simple demonstration with the mock handler."""
    print("\n=== Simple Multi-Model Processor Demo ===\n")
    
    # Create a processor with mock components
    processor = ProcessorEngine(
        config={
            "processor": {
                "default_model": "mock-model",
                "enable_caching": False
            },
            "models": {
                "mock-model": {
                    "handler_type": "mock",
                    "latency": 0.5  # Reduced latency for demo
                }
            }
        }
    )
    
    # Define some sample tasks
    tasks = [
        "Summarize the key features of the Multi-Model Processor component.",
        "What are the benefits of using the ProcessorState to track tasks?",
        "How does the error handling framework improve system reliability?",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}: {task} ---\n")
        
        # Process the task
        start_time = time.time()
        result = await processor.process_task(
            task=task,
            context={"demo": True, "task_number": i}
        )
        processing_time = time.time() - start_time
        
        # Display the results
        print(f"Output from model '{result['model']}':")
        print(f"{result['output']}\n")
        
        # Display some metrics
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Tokens: {result['tokens'].get('prompt', 0)} input, "
              f"{result['tokens'].get('completion', 0)} output")
        
        # Show processing stages
        if "state" in result:
            stages = [stage_info["stage"] for stage_info in result["state"]["stage_history"]]
            print(f"Processing stages: {' → '.join(stages)}")
        
        # Wait between tasks
        if i < len(tasks):
            print("\nWaiting for next task...")
            await asyncio.sleep(1)
    
    print("\n=== Demo Complete ===")


async def run_processor_with_custom_handler():
    """
    Run a demonstration with a custom handler.
    
    This requires the OPENAI_API_KEY environment variable to be set.
    If not available, it falls back to the mock handler.
    """
    print("\n=== Advanced Multi-Model Processor Demo ===\n")
    
    # Try to get OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Configure handlers based on available API keys
    models_config = {
        "mock-model": {
            "handler_type": "mock",
            "latency": 0.5
        }
    }
    
    # Add OpenAI configuration if API key is available
    handler_available = "Mock"
    if api_key:
        models_config["openai-gpt"] = {
            "handler_type": "openai",
            "api_key": api_key,
            "default_model": "gpt-3.5-turbo"
        }
        handler_available = "OpenAI"
    
    print(f"Using {handler_available} handler for demonstration")
    
    # Create the processor
    processor = ProcessorEngine(
        config={
            "processor": {
                "default_model": "openai-gpt" if api_key else "mock-model",
                "enable_caching": False
            },
            "models": models_config
        }
    )
    
    # Complex task with context
    task = """
    Analyze the benefits of implementing a Multi-Model Processor for language model applications.
    Consider factors like model selection, error handling, and integration with other components.
    """
    
    context = {
        "user_preferences": {
            "detail_level": "high",
            "format": "markdown"
        },
        "project_info": {
            "name": "PTOLEMY",
            "description": "Knowledge management and reasoning system"
        }
    }
    
    print(f"Processing complex task with {handler_available} handler...\n")
    print(f"Task: {task.strip()}\n")
    
    # Process the task
    start_time = time.time()
    result = await processor.process_task(
        task=task,
        context=context
    )
    processing_time = time.time() - start_time
    
    # Display the results
    print(f"\nOutput from model '{result['model']}':")
    print(f"{result['output']}\n")
    
    # Display metrics and state information
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Tokens: {result['tokens'].get('prompt', 0)} input, "
          f"{result['tokens'].get('completion', 0)} output")
    
    # Get available models
    models = await processor.get_available_models()
    print(f"\nAvailable models: {', '.join([model['id'] for model in models])}")
    
    print("\n=== Advanced Demo Complete ===")


async def main():
    """Main function to run all demos."""
    # Print header
    print("\n" + "="*50)
    print("PTOLEMY Multi-Model Processor Demonstration")
    print("="*50)
    
    try:
        # Run the simple demo
        await run_simple_demo()
        
        # Run the advanced demo with custom handler if available
        await run_processor_with_custom_handler()
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nDemonstration completed.")


if __name__ == "__main__":
    asyncio.run(main())
