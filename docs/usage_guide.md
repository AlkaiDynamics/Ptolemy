# PTOLEMY Usage Guide

This guide provides detailed instructions for setting up and using the PTOLEMY system with multiple AI providers.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Available AI Providers](#available-ai-providers)
4. [Basic Usage](#basic-usage)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ptolemy
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install optional AI provider packages based on your needs:
   ```bash
   # For OpenAI (required)
   pip install openai

   # For Anthropic
   pip install anthropic

   # For Mistral AI
   pip install mistralai

   # For Groq
   pip install groq

   # For Cohere
   pip install cohere

   # For Google's Generative AI
   pip install google-generativeai

   # For DeepSeek
   pip install deepseek

   # For OpenRouter
   pip install openrouter
   ```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with your API keys and default settings:

```
# Required
OPENAI_API_KEY=your_openai_api_key
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4
LOG_LEVEL=info

# Optional - Other Providers
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_DEFAULT_MODEL=claude-3-sonnet-20240229

MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_DEFAULT_MODEL=mistral-medium

GROQ_API_KEY=your_groq_api_key
GROQ_DEFAULT_MODEL=mixtral-8x7b-32768

COHERE_API_KEY=your_cohere_api_key
COHERE_DEFAULT_MODEL=command-r

GOOGLE_API_KEY=your_google_api_key
GOOGLE_DEFAULT_MODEL=gemini-pro

DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_DEFAULT_MODEL=deepseek-chat

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_DEFAULT_MODEL=openai/gpt-4-turbo

XAI_API_KEY=your_xai_api_key
XAI_DEFAULT_MODEL=xai-chat

QWEN_API_KEY=your_qwen_api_key
QWEN_DEFAULT_MODEL=qwen-max

MICROSOFT_API_KEY=your_microsoft_api_key
MICROSOFT_ENDPOINT=your_azure_endpoint
MICROSOFT_DEFAULT_MODEL=azure-gpt-4

# Local Models
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_DEFAULT_MODEL=mixtral

LMSTUDIO_API_BASE=http://localhost:1234/v1
LMSTUDIO_DEFAULT_MODEL=local-model
```

### Directory Structure

Ensure the following directories exist:
```
ptolemy/
├── data/
│   ├── context/
│   │   ├── relationships/
│   │   ├── patterns/
│   │   └── insights/
│   └── temporal/
└── logs/
```

## Available AI Providers

PTOLEMY supports the following AI providers:

| Provider | Description | Models | Package |
|----------|-------------|--------|---------|
| OpenAI | Primary provider | gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo | `openai` |
| Anthropic | Claude models | claude-3-opus, claude-3-sonnet, claude-3-haiku | `anthropic` |
| Mistral | Mistral models | mistral-tiny, mistral-small, mistral-medium, mistral-large | `mistralai` |
| Groq | Fast inference | llama2-70b, mixtral-8x7b, gemma-7b | `groq` |
| Cohere | Command models | command, command-light, command-r, command-r-plus | `cohere` |
| Google | Gemini models | gemini-pro, gemini-pro-vision, gemini-ultra | `google-generativeai` |
| DeepSeek | DeepSeek models | deepseek-coder, deepseek-chat | `deepseek` |
| OpenRouter | Multi-provider router | Various models from different providers | `openrouter` |
| Ollama | Local models | llama2, mistral, mixtral, phi, gemma, etc. | Uses OpenAI client |
| LM Studio | Local models | Any model loaded in LM Studio | Uses OpenAI client |
| Microsoft Azure | Azure OpenAI | azure-gpt-4, azure-gpt-35-turbo | `openai` with Azure config |
| XAI | XAI models | xai-chat, xai-instruct | Not yet available |
| Qwen | Qwen models | qwen-max, qwen-plus, qwen-turbo | Not yet available |

## Basic Usage

### Command Line Interface

PTOLEMY provides a command-line interface for common operations:

```bash
# Initialize the system
python cli.py init

# Generate content with the default provider
python cli.py generate "Create a Python function to calculate Fibonacci numbers"

# Specify a provider and model
python cli.py generate "Create a Python function to calculate Fibonacci numbers" --provider anthropic --model claude-3-sonnet-20240229

# Run a multi-stage prompt chain
python cli.py chain "Design a REST API for a blog system"

# Provide feedback
python cli.py feedback "I prefer more detailed comments in the code" --type user_preference

# Analyze feedback trends
python cli.py analyze
```

### Python API

You can also use PTOLEMY programmatically:

```python
import asyncio
from ptolemy.context_engine import ContextEngine
from ptolemy.multi_model import MultiModelProcessor

async def main():
    # Initialize components
    context_engine = ContextEngine()
    multi_model = MultiModelProcessor(context_engine)
    
    # Simple task
    result = await multi_model.route_task(
        "Create a Python function to calculate Fibonacci numbers",
        "implementer",
        options={"provider": "openai", "model": "gpt-4"}
    )
    print(result)
    
    # Multi-stage task
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
    
    for i, result in enumerate(results):
        print(f"Stage {i+1} Result:")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Custom Model Registration

You can register custom model configurations:

```python
multi_model.register_model("code_reviewer", {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "system_prompt": "You are a code reviewer. Analyze the code for bugs, security issues, and performance problems.",
    "temperature": 0.2
})

result = await multi_model.route_task(
    "Review this code: def add(a, b): return a + b",
    "code_reviewer"
)
```

### Using Multiple Providers in Sequence

For complex tasks, you can chain multiple providers:

```python
stages = [
    {
        "model_type": "planner",
        "name": "Planning Stage",
        "options": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "additional_instructions": "Focus on high-level architecture"
        },
        "update_task": True
    },
    {
        "model_type": "implementer",
        "name": "Implementation Stage",
        "options": {
            "provider": "openai",
            "model": "gpt-4",
            "additional_instructions": "Implement the design from the planning stage"
        }
    },
    {
        "model_type": "tester",
        "name": "Testing Stage",
        "options": {
            "provider": "mistral",
            "model": "mistral-medium",
            "additional_instructions": "Create unit tests for the implementation"
        }
    }
]

results = await multi_model.route_multi_stage(
    "Create a REST API for user authentication with JWT",
    stages
)
```

### Using Local Models

For privacy or cost reasons, you can use local models with Ollama or LM Studio:

```python
# Using Ollama
result = await multi_model.route_task(
    "Explain quantum computing in simple terms",
    "explainer",
    options={"provider": "ollama", "model": "mixtral"}
)

# Using LM Studio
result = await multi_model.route_task(
    "Write a short poem about AI",
    "creative",
    options={"provider": "lmstudio", "model": "local-model"}
)
```

## API Reference

### MultiModelProcessor

The core class for routing tasks to different AI providers.

#### Methods:

- `initialize_clients()`: Initialize AI clients for all configured providers
- `register_model(model_type, config)`: Register a new model type with configuration
- `route_task(task, model_type, options=None)`: Route a task to a specific model type
- `route_multi_stage(task, stages, options=None)`: Route a task through multiple stages of models

#### Options Dictionary:

- `provider`: The AI provider to use (e.g., "openai", "anthropic")
- `model`: The specific model to use
- `additional_instructions`: Extra instructions to append to the prompt
- `temperature`: Sampling temperature (0.0 to 1.0)

### ContextEngine

Manages context for AI models.

#### Methods:

- `get_model_context(task)`: Get relevant context for a task
- `add_relationship(source, target, relationship_type)`: Add a relationship between entities
- `add_pattern(pattern_name, pattern_data)`: Add a reusable implementation pattern
- `add_insight(insight_name, insight_data)`: Add a project-specific insight

## Troubleshooting

### Common Issues

#### API Key Issues

If you encounter authentication errors:
- Verify that your API keys in the `.env` file are correct
- Check if your API keys have expired or reached usage limits
- Ensure the environment variables are properly loaded

#### Package Installation Problems

If you have issues with package installation:
- Try updating pip: `pip install --upgrade pip`
- Install packages individually to identify problematic dependencies
- Check for version conflicts in your requirements

#### Provider Not Available

If a provider is not available:
- Check if you've installed the required package
- Verify that the API key is set in the `.env` file
- Some providers (like XAI and Qwen) may not be available yet

#### Model Errors

If you encounter model-specific errors:
- Verify that the model name is correct
- Check if the model is available for your API tier
- Some models may have context length limitations

### Getting Help

If you continue to experience issues:
- Check the logs in the `logs/` directory
- Consult the provider's documentation for specific API requirements
- Open an issue in the project repository with detailed error information
