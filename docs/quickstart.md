# PTOLEMY Quickstart Guide

This guide will help you get started with PTOLEMY quickly, focusing on the Python implementation.

## Installation

1. **Set up a Python virtual environment**:
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your environment**:
   Create a `.env` file in the root directory with at least the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DEFAULT_PROVIDER=openai
   DEFAULT_MODEL=gpt-4
   LOG_LEVEL=info
   ```

## Basic Usage

### Command Line Interface

```bash
# Initialize the system
python cli.py init

# Generate content
python cli.py generate "Create a Python function to calculate Fibonacci numbers"

# Specify a provider and model
python cli.py generate "Write a blog post about AI" --provider anthropic --model claude-3-sonnet-20240229
```

### Python API

```python
import asyncio
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.context_engine import ContextEngine

async def main():
    # Initialize components
    context_engine = ContextEngine()
    multi_model = MultiModelProcessor(context_engine)
    
    # Simple task
    result = await multi_model.route_task(
        "Create a Python function to calculate Fibonacci numbers",
        "implementer"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Available AI Providers

PTOLEMY supports multiple AI providers:

- **OpenAI** (default): GPT-4, GPT-3.5-Turbo
- **Anthropic**: Claude models
- **Mistral AI**: Mistral models
- **Groq**: Fast inference
- **Cohere**: Command models
- **Google**: Gemini models
- **DeepSeek**: DeepSeek models
- **OpenRouter**: Multi-provider router
- **Microsoft Azure**: Azure OpenAI
- **Local models** via Ollama or LM Studio

## Next Steps

- Read the full [Usage Guide](usage_guide.md) for detailed instructions
- Check the [Provider Guide](provider_guide.md) for provider-specific information
- Explore the [API Reference](api_reference.md) for advanced usage
