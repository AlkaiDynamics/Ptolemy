#!/usr/bin/env python
# test_latent_reasoning.py
import asyncio
from loguru import logger
from pathlib import Path

from ptolemy.context_engine.core import ContextEngine
from ptolemy.latent_reasoning import LatentReasoningEngine
from ptolemy.temporal_core import TemporalCore


async def main():
    # Set up logging
    logger.info("Starting Latent Reasoning Module test")
    
    # Initialize components
    logger.info("Initializing TemporalCore and ContextEngine...")
    temporal_core = TemporalCore()
    await temporal_core.initialize()
    
    context_engine = ContextEngine(temporal_core)
    await context_engine.initialize()
    
    logger.info("Initializing LatentReasoningEngine...")
    latent_reasoning = LatentReasoningEngine(context_engine)
    
    # Simple test tasks of varying complexity
    tasks = [
        "Create a basic user authentication module",
        "Implement a recommendation system for an e-commerce platform based on user purchase history and browsing behavior",
        "Design an error handling middleware that logs errors, notifies developers, and provides user-friendly responses",
        "Refactor our codebase to use async/await pattern for all I/O operations"
    ]
    
    # Process each task with different iteration settings
    for task in tasks:
        logger.info(f"\nTesting task: {task}")
        
        # Test with different iteration settings
        for iterations in [4, 8, 16]:
            logger.info(f"Processing with {iterations} iterations...")
            result = await latent_reasoning.process(
                task=task, 
                iterations=iterations,
                adaptive=True,
                task_metadata={"type": "code_generation", "include_trajectory": True}
            )
            
            # Log results - handle potential errors
            logger.info(f"Process completed in {result.get('time', 0):.4f} seconds")
            
            if "error" in result:
                logger.error(f"Error encountered: {result['error']}")
                continue
                
            logger.info(f"Iterations used: {result.get('iterations', 0)} (adaptive: {result.get('adaptive_stopped', False)})")
            
            if "output" in result and "key_concepts" in result["output"]:
                logger.info(f"Key concepts: {result['output']['key_concepts']}")
                
                attention_focus = result['output'].get('attention_focus')
                if attention_focus:
                    logger.info("Focus areas:")
                    for area, weight in attention_focus.items():
                        logger.info(f"  - {area}: {weight:.2f}")
    
    # Test task complexity analysis
    logger.info("\nTesting task complexity analysis...")
    for task in tasks:
        complexity = await latent_reasoning.analyze_task_complexity(task, "Sample context")
        logger.info(f"Task: {task}")
        logger.info(f"  Complexity score: {complexity.get('complexity_score', 0):.2f}")
        logger.info(f"  Recommended iterations: {complexity.get('recommended_iterations', 0)}")
    
    # Print metrics summary
    logger.info("\nMetrics Summary:")
    metrics = latent_reasoning.get_metrics_summary()
    logger.info(f"Total processes: {metrics.get('total_processes', 0)}")
    logger.info(f"Average iterations: {metrics.get('avg_iterations_per_process', 0):.2f}")
    logger.info(f"Average time per process: {metrics.get('avg_time_per_process', 0):.4f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
