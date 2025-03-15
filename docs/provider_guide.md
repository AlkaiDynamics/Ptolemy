# AI Provider Integration Guide for PTOLEMY

This guide provides detailed instructions for integrating and using different AI providers with the PTOLEMY system.

## Table of Contents

1. [Provider Overview](#provider-overview)
2. [Integration Details](#integration-details)
3. [Configuration Examples](#configuration-examples)
4. [Fallback Mechanisms](#fallback-mechanisms)
5. [Best Practices](#best-practices)

## Provider Overview

PTOLEMY supports multiple AI providers to give you flexibility in choosing the best model for your specific tasks. Each provider has different strengths, pricing models, and available features.

### Core Providers

| Provider | Strengths | Use Cases | API Base |
|----------|-----------|-----------|----------|
| OpenAI | General purpose, strong coding | Code generation, planning, reasoning | `https://api.openai.com/v1` |
| Anthropic | Safety, long context, reasoning | Planning, analysis, content creation | `https://api.anthropic.com/v1` |
| Mistral | Efficiency, multilingual | Translation, summarization, Q&A | `https://api.mistral.ai/v1` |
| Groq | Speed, low latency | Real-time applications, chat | `https://api.groq.com/openai/v1` |
| Cohere | Embeddings, RAG, search | Document analysis, semantic search | `https://api.cohere.ai/v1` |
| Google | Multimodal, research-focused | Image analysis, complex reasoning | Via SDK |
| DeepSeek | Code generation, technical tasks | Programming, technical documentation | `https://api.deepseek.com/v1` |

### Additional Providers

| Provider | Status | Notes |
|----------|--------|-------|
| OpenRouter | Available | Meta-provider for accessing multiple APIs |
| Microsoft Azure | Available | Enterprise-grade with compliance features |
| XAI | Planned | Package not yet available |
| Qwen | Planned | Package not yet available |
| Ollama | Available | Local deployment for privacy and cost savings |
| LM Studio | Available | Local deployment with UI for model management |

## Integration Details

### How Provider Integration Works

The PTOLEMY system integrates providers through the `multi_model.py` module, which:

1. Initializes clients for each configured provider
2. Handles authentication and API configuration
3. Routes tasks to the appropriate provider based on user preferences
4. Provides fallback mechanisms when providers are unavailable

### Client Initialization

The system attempts to initialize clients for all configured providers during startup:

```python
def initialize_clients(self):
    for provider, config in AI_PROVIDERS.items():
        if provider == "openai" and config["api_key"]:
            self.ai_clients[provider] = openai.OpenAI(api_key=config["api_key"])
        elif provider == "anthropic" and config["api_key"]:
            try:
                import anthropic
                self.ai_clients[provider] = anthropic.Anthropic(api_key=config["api_key"])
            except ImportError:
                logger.warning("Anthropic package not installed. Skipping client initialization.")
        # Additional providers...
```

### Task Routing

Tasks are routed to providers based on the specified model type and options:

```python
async def route_task(self, task, model_type, options=None):
    options = options or {}
    provider = options.get("provider", self.default_provider)
    model = options.get("model", self.get_default_model(provider))
    
    # Get the client for the specified provider
    client = self.ai_clients.get(provider)
    
    if not client:
        logger.warning(f"Provider {provider} not available. Using fallback provider.")
        return self.get_mock_response(task, model_type)
    
    # Route to the appropriate provider-specific method
    if provider == "openai":
        return await self.openai_task(client, task, model, model_type, options)
    elif provider == "anthropic":
        return await self.anthropic_task(client, task, model, model_type, options)
    # Additional providers...
```

## Configuration Examples

### OpenAI Configuration

```python
# .env file
OPENAI_API_KEY=your_openai_api_key
DEFAULT_MODEL=gpt-4

# Usage example
result = await multi_model.route_task(
    "Create a Python function to calculate Fibonacci numbers",
    "implementer",
    options={"provider": "openai", "model": "gpt-4"}
)
```

### Anthropic Configuration

```python
# .env file
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_DEFAULT_MODEL=claude-3-sonnet-20240229

# Usage example
result = await multi_model.route_task(
    "Write a detailed analysis of quantum computing",
    "researcher",
    options={"provider": "anthropic", "model": "claude-3-opus-20240229"}
)
```

### Mistral Configuration

```python
# .env file
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_DEFAULT_MODEL=mistral-medium

# Usage example
result = await multi_model.route_task(
    "Summarize this research paper",
    "summarizer",
    options={"provider": "mistral", "model": "mistral-large"}
)
```

### DeepSeek Configuration

```python
# .env file
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_DEFAULT_MODEL=deepseek-chat

# Usage example
result = await multi_model.route_task(
    "Generate a complex algorithm for graph traversal",
    "coder",
    options={"provider": "deepseek", "model": "deepseek-coder"}
)
```

### Azure OpenAI Configuration

```python
# .env file
MICROSOFT_API_KEY=your_azure_api_key
MICROSOFT_ENDPOINT=https://your-resource-name.openai.azure.com
MICROSOFT_DEFAULT_MODEL=azure-gpt-4

# Usage example
result = await multi_model.route_task(
    "Create a compliance report template",
    "writer",
    options={"provider": "microsoft", "model": "azure-gpt-4"}
)
```

### Local Model Configuration (Ollama)

```python
# .env file
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_DEFAULT_MODEL=mixtral

# Usage example
result = await multi_model.route_task(
    "Explain how transistors work",
    "explainer",
    options={"provider": "ollama", "model": "llama2"}
)
```

## Fallback Mechanisms

PTOLEMY includes robust fallback mechanisms to handle cases where providers or packages are unavailable:

### Package Unavailability

If a required package is not installed, the system will:
1. Log a warning
2. Skip client initialization for that provider
3. Use a fallback provider when tasks are routed to the unavailable provider

```python
try:
    import anthropic
    self.ai_clients[provider] = anthropic.Anthropic(api_key=config["api_key"])
except ImportError:
    logger.warning("Anthropic package not installed. Skipping client initialization.")
```

### API Key Missing

If an API key is not configured, the system will:
1. Skip client initialization for that provider
2. Log a warning if a task is routed to the provider

```python
if provider == "openai" and config["api_key"]:
    self.ai_clients[provider] = openai.OpenAI(api_key=config["api_key"])
else:
    logger.warning(f"API key for {provider} not configured.")
```

### Mock Responses

For providers that are not yet available (like XAI and Qwen), the system provides mock responses:

```python
def get_mock_response(self, task, model_type):
    logger.warning(f"Using mock response for {model_type}")
    return f"Mock response for task: {task}\nModel type: {model_type}\nNote: This provider is not currently available."
```

## Best Practices

### Provider Selection

Choose the right provider for each task type:

- **OpenAI**: Best for code generation and general-purpose tasks
- **Anthropic**: Excellent for safety-critical applications and long-context reasoning
- **Mistral**: Good for efficient, cost-effective processing
- **Groq**: Ideal for low-latency applications
- **DeepSeek**: Specialized for code generation and technical tasks
- **Local models**: Best for privacy-sensitive applications or offline use

### Cost Optimization

To optimize costs:

1. Use smaller models for simpler tasks
2. Implement caching for repeated queries
3. Consider local models for development and testing
4. Set appropriate context lengths to avoid unnecessary token usage

```python
# Example of cost-optimized configuration
stages = [
    {
        "model_type": "planner",
        "name": "Planning Stage",
        "options": {
            "provider": "mistral",  # Cost-effective for planning
            "model": "mistral-small"
        }
    },
    {
        "model_type": "implementer",
        "name": "Implementation Stage",
        "options": {
            "provider": "openai",  # Best quality for implementation
            "model": "gpt-4"
        }
    }
]
```

### Error Handling

Implement proper error handling for API calls:

```python
try:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content
except Exception as e:
    logger.error(f"Error calling {provider} API: {str(e)}")
    # Use fallback provider
    return await self.route_task(task, model_type, {"provider": "fallback_provider"})
```

### Security Considerations

Protect your API keys and sensitive data:

1. Store API keys in environment variables, never in code
2. Use Azure OpenAI for enterprise-grade security compliance
3. Consider local models for processing sensitive information
4. Implement rate limiting to prevent excessive API usage

### Testing New Providers

When integrating a new provider:

1. Start with simple, well-defined tasks
2. Compare outputs with established providers
3. Gradually increase complexity and task diversity
4. Monitor costs and performance metrics
5. Implement proper error handling before production use
