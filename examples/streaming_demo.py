"""
Streaming Demo for Multi-Model Processor

This script demonstrates the streaming capability of the Multi-Model Processor
by processing a task with real-time streaming responses.
"""

import asyncio
import os
import sys
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ptolemy.multi_model_processor import ProcessorEngine
from ptolemy.multi_model_processor.handlers.base_handler import BaseModelHandler
from ptolemy.multi_model_processor.handlers.anthropic_handler import AnthropicModelHandler

# Create a simple mock handler for testing if no API keys are available
class MockHandler(BaseModelHandler):
    """Mock handler for demonstration purposes."""
    
    async def process(self, prompt: str, parameters: dict = None) -> dict:
        """Process a prompt with mock responses."""
        return {
            "text": f"Mock response to: {prompt[:30]}...",
            "model": "mock-model",
            "tokens": {"prompt": len(prompt.split()), "completion": 20}
        }
    
    async def get_capabilities(self) -> dict:
        """Get capabilities of this model."""
        return {
            "max_tokens": 1000,
            "supports_streaming": True,
            "strengths": ["testing", "demo"]
        }
    
    async def process_stream(self, prompt: str, parameters: dict = None) -> dict:
        """Process a prompt with streaming."""
        # Simulate thinking time
        await asyncio.sleep(0.5)
        
        # Generate a fake response word by word
        response = f"This is a simulated streaming response for the prompt: {prompt[:20]}..."
        words = response.split()
        
        for i, word in enumerate(words):
            # Simulate network delay
            await asyncio.sleep(0.2)
            
            # Yield each word as a chunk
            yield {
                "text": word + " ",
                "finished": i == len(words) - 1,
                "model": "mock-model",
                "tokens": {"prompt": 0, "completion": 1} if i > 0 else {"prompt": len(prompt.split()), "completion": 0}
            }


async def main():
    """Demo the streaming capability of the Multi-Model Processor."""
    # Initialize processor
    processor = ProcessorEngine()
    
    # Register handlers based on available API keys
    handlers_registered = False
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        processor.register_handler("anthropic", AnthropicModelHandler({
            "api_key": os.environ["ANTHROPIC_API_KEY"],
            "model_id": "claude-3-haiku-20240307"
        }))
        logger.info("Registered Anthropic handler")
        handlers_registered = True
    
    # Always register mock handler as fallback
    processor.register_handler("mock", MockHandler())
    logger.info("Registered Mock handler as fallback")
    handlers_registered = True
    
    if not handlers_registered:
        logger.warning("No handlers registered. Using built-in mock handler.")
    
    # Example task
    task = "Write a short poem about artificial intelligence in 4 lines."
    print(f"\nTask: {task}\n")
    print("Response:")
    
    # Process with streaming
    started_output = False
    async for chunk in processor.process_task_stream(task):
        if "event" in chunk:
            # Event notification
            if chunk["event"] == "reasoning_complete":
                print("\n[Reasoning complete]")
        elif "error" in chunk:
            # Error occurred
            print(f"\nError: {chunk['error']}")
        elif "text" in chunk:
            # Print text without newline to simulate streaming
            print(chunk["text"], end="", flush=True)
            started_output = True
            
            # Print newline after completion
            if chunk.get("finished", False) and started_output:
                print("\n")
                
                # Print information about the processing
                if "state" in chunk:
                    state = chunk["state"]
                    print(f"\nProcessed with: {state.get('selected_model', 'unknown')}")
                    print(f"Reasoning iterations: {state.get('reasoning_iterations', 0)}")
                    if "token_usage" in state:
                        print(f"Tokens: {state['token_usage']}")

if __name__ == "__main__":
    asyncio.run(main())
