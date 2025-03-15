# Multi-Model Processor

## Overview

The Multi-Model Processor is a core component of the PTOLEMY system that provides intelligent task processing across multiple language models. It offers a unified interface for task processing, model selection, and integration with the Context Engine and Latent Reasoning modules.

## Key Features

- **Model Abstraction**: Interfaces with multiple language models through a unified API
- **Intelligent Routing**: Selects the optimal model for each task based on capabilities and requirements
- **Context Integration**: Integrates with the PTOLEMY Context Engine to enhance task processing with relevant context
- **Latent Reasoning**: Leverages the Latent Reasoning module to improve task understanding
- **Comprehensive State Tracking**: Tracks the complete processing lifecycle with detailed metrics
- **Robust Error Handling**: Provides graceful degradation with fallbacks when components fail

## Architecture

The Multi-Model Processor is organized into the following components:

### Core Components

- **ProcessorEngine**: Central orchestrator that manages the processing workflow
- **ProcessorState**: State container that tracks the complete processing lifecycle
- **BaseModelHandler**: Abstract interface for model-specific handlers

### Utilities

- **ConfigManager**: Manages configuration with support for file and environment overrides
- **ComponentLoader**: Handles dynamic loading of components with fallback support
- **ErrorHandler**: Centralizes error handling and logging

### Handlers

Model-specific handlers that implement the BaseModelHandler interface:
- **MockModelHandler**: Test implementation for development and testing

## Usage Example

```python
import asyncio
from ptolemy.multi_model_processor.processor import ProcessorEngine
from ptolemy.latent_reasoning.engine import LatentReasoningEngine
from ptolemy.context_engine import ContextEngine

async def process_task():
    # Initialize engines
    latent_reasoning = LatentReasoningEngine(...)
    context_engine = ContextEngine(...)
    
    # Initialize processor with components
    processor = ProcessorEngine(
        latent_reasoning_engine=latent_reasoning,
        context_engine=context_engine,
        config_path="config.json"
    )
    
    # Process a task
    result = await processor.process_task(
        task="Analyze the performance implications of increasing the cache size.",
        context={"user_id": "user123", "project": "database-optimization"},
        model_preference="gpt-4"  # Optional model preference
    )
    
    # Use the result
    print(f"Output: {result['output']}")
    print(f"Model used: {result['model']}")
    print(f"Token usage: {result['tokens']}")
    
    # Access detailed processing state
    state = result["state"]
    print(f"Processing stages: {[s['stage'] for s in state['stage_history']]}")
    print(f"Processing time: {result['time']:.2f}s")

# Run the async function
asyncio.run(process_task())
```

## Extending the Processor

### Adding a New Model Handler

To add support for a new model:

1. Create a new handler class that extends `BaseModelHandler`
2. Implement the required methods: `process()` and `get_capabilities()`
3. Register your handler in the configuration:

```json
{
  "models": {
    "my-new-model": {
      "handler_type": "my_new",
      "api_key": "${MY_API_KEY}",
      "capabilities": {
        "max_tokens": 4096,
        "supports_streaming": true
      }
    }
  }
}
```

## Error Handling

The Multi-Model Processor includes a comprehensive error handling system:

- **Error Severity Levels**: INFO, WARNING, ERROR, CRITICAL
- **Contextual Error Information**: Errors include relevant context for debugging
- **Graceful Degradation**: Falls back to simpler processing when components fail
- **Detailed Error Tracking**: All errors are recorded in the processor state

## Performance Monitoring

The processor tracks detailed metrics for each processing task:

- **Token Usage**: Input and output tokens for each model call
- **Processing Time**: Total and per-stage timing information
- **Reasoning Iterations**: Number of latent reasoning iterations performed
- **Model Selection**: Model selection decisions and reasoning

These metrics can be accessed via the `get_metrics()` method on the ProcessorEngine.
