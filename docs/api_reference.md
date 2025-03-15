# PTOLEMY API Reference

This document provides a comprehensive reference for the PTOLEMY Python API.

## Table of Contents

1. [MultiModelProcessor](#multimodelprocessor)
2. [ContextEngine](#contextengine)
3. [Configuration](#configuration)
4. [Utility Functions](#utility-functions)

## MultiModelProcessor

The core class for routing tasks to different AI providers.

### Initialization

```python
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.context_engine import ContextEngine

context_engine = ContextEngine()
multi_model = MultiModelProcessor(context_engine)
```

### Methods

#### initialize_clients()

Initializes AI clients for all configured providers.

```python
multi_model.initialize_clients()
```

#### register_model(model_type, config)

Registers a new model type with configuration.

```python
multi_model.register_model("code_reviewer", {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "system_prompt": "You are a code reviewer...",
    "temperature": 0.2
})
```

#### async route_task(task, model_type, options=None)

Routes a task to a specific model type.

```python
result = await multi_model.route_task(
    "Create a Python function to calculate Fibonacci numbers",
    "implementer",
    options={
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "additional_instructions": "Include detailed comments"
    }
)
```

**Parameters:**
- `task` (str): The task description or prompt
- `model_type` (str): The type of model to use (e.g., "implementer", "planner")
- `options` (dict, optional): Additional options for the task

**Returns:**
- `str`: The generated response

#### async route_multi_stage(task, stages, options=None)

Routes a task through multiple stages of models.

```python
stages = [
    {
        "model_type": "planner",
        "name": "Planning Stage",
        "options": {"provider": "anthropic"}
    },
    {
        "model_type": "implementer",
        "name": "Implementation Stage",
        "options": {"provider": "openai"}
    }
]

results = await multi_model.route_multi_stage(
    "Design a REST API for a blog system",
    stages
)
```

**Parameters:**
- `task` (str): The task description or prompt
- `stages` (list): A list of stage configurations
- `options` (dict, optional): Global options for all stages

**Returns:**
- `list`: The generated responses for each stage

#### get_default_model(provider)

Gets the default model for a provider.

```python
default_model = multi_model.get_default_model("openai")
```

**Parameters:**
- `provider` (str): The provider name

**Returns:**
- `str`: The default model name

## ContextEngine

Manages context for AI models.

### Initialization

```python
from ptolemy.context_engine import ContextEngine

context_engine = ContextEngine()
```

### Methods

#### get_model_context(task)

Gets relevant context for a task.

```python
context = context_engine.get_model_context("Create a login system")
```

**Parameters:**
- `task` (str): The task description

**Returns:**
- `dict`: Relevant context for the task

#### add_relationship(source, target, relationship_type)

Adds a relationship between entities.

```python
context_engine.add_relationship(
    "User", 
    "Authentication", 
    "requires"
)
```

**Parameters:**
- `source` (str): The source entity
- `target` (str): The target entity
- `relationship_type` (str): The type of relationship

#### add_pattern(pattern_name, pattern_data)

Adds a reusable implementation pattern.

```python
context_engine.add_pattern(
    "repository_pattern",
    {
        "description": "Data access pattern that separates business logic from data access",
        "example": "class UserRepository: ...",
        "use_cases": ["Database access", "API integration"]
    }
)
```

**Parameters:**
- `pattern_name` (str): The name of the pattern
- `pattern_data` (dict): The pattern data

#### add_insight(insight_name, insight_data)

Adds a project-specific insight.

```python
context_engine.add_insight(
    "authentication_flow",
    {
        "description": "The project uses JWT for authentication",
        "details": "Tokens expire after 24 hours",
        "related_components": ["AuthService", "UserController"]
    }
)
```

**Parameters:**
- `insight_name` (str): The name of the insight
- `insight_data` (dict): The insight data

## Configuration

The `config.py` module contains configuration settings for the PTOLEMY system.

### AI_PROVIDERS

Dictionary containing configuration for all supported AI providers.

```python
from ptolemy.config import AI_PROVIDERS

# Access provider configuration
openai_config = AI_PROVIDERS["openai"]
```

### MODEL_TYPES

Dictionary containing configuration for different model types.

```python
from ptolemy.config import MODEL_TYPES

# Access model type configuration
implementer_config = MODEL_TYPES["implementer"]
```

### Environment Variables

Environment variables are loaded from the `.env` file and accessible through the `os.getenv` function.

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
default_model = os.getenv("DEFAULT_MODEL")
```

## Utility Functions

### logging

The PTOLEMY system uses Python's logging module for logging.

```python
import logging

logger = logging.getLogger(__name__)
logger.info("This is an informational message")
logger.warning("This is a warning message")
logger.error("This is an error message")
```

### async_timeout

Utility for adding timeouts to asynchronous operations.

```python
import asyncio
from async_timeout import timeout

async def call_with_timeout():
    async with timeout(10):  # 10 second timeout
        result = await some_async_function()
    return result
```
