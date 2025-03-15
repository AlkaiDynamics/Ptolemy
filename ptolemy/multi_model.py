import json
import importlib
import os
from typing import Dict, List, Optional, Any, Union

import openai
import requests
from loguru import logger

from ptolemy.config import AI_PROVIDERS, MODEL_REGISTRY, DEFAULT_PROVIDER
from ptolemy.context_engine import ContextEngine


class MultiModelProcessor:
    """
    Multi-Model Processor delegates tasks to specialized AI models,
    integrates independently generated components, and ensures consistency
    across software modules.
    """
    
    def __init__(self, context_engine: ContextEngine):
        self.context_engine = context_engine
        self.model_registry = MODEL_REGISTRY.copy()
        self.ai_clients = {}
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize AI clients for all configured providers."""
        for provider, config in AI_PROVIDERS.items():
            try:
                if provider == "openai" and config["api_key"]:
                    self.ai_clients[provider] = openai.OpenAI(api_key=config["api_key"])
                    logger.info(f"Initialized {provider} client")
                elif provider == "anthropic" and config["api_key"]:
                    # Import anthropic only if needed
                    try:
                        import anthropic
                        self.ai_clients[provider] = anthropic.Anthropic(api_key=config["api_key"])
                        logger.info(f"Initialized {provider} client")
                    except ImportError:
                        logger.warning(f"Anthropic package not installed. To use Anthropic models, install with: pip install anthropic")
                elif provider == "mistral" and config["api_key"]:
                    # Import mistralai only if needed
                    try:
                        import mistralai.client
                        self.ai_clients[provider] = mistralai.client.MistralClient(api_key=config["api_key"])
                        logger.info(f"Initialized {provider} client")
                    except ImportError:
                        logger.warning(f"Mistral package not installed. To use Mistral models, install with: pip install mistralai")
                elif provider == "groq" and config["api_key"]:
                    # Groq uses OpenAI-compatible API
                    self.ai_clients[provider] = openai.OpenAI(
                        api_key=config["api_key"],
                        base_url="https://api.groq.com/openai/v1"
                    )
                    logger.info(f"Initialized {provider} client")
                elif provider == "cohere" and config["api_key"]:
                    # Import cohere only if needed
                    try:
                        import cohere
                        self.ai_clients[provider] = cohere.Client(api_key=config["api_key"])
                        logger.info(f"Initialized {provider} client")
                    except ImportError:
                        logger.warning(f"Cohere package not installed. To use Cohere models, install with: pip install cohere")
                elif provider == "ollama" and config["api_base"]:
                    # Ollama uses OpenAI-compatible API
                    self.ai_clients[provider] = openai.OpenAI(base_url=config["api_base"])
                    logger.info(f"Initialized {provider} client")
                elif provider == "lmstudio" and config["api_base"]:
                    # LM Studio uses OpenAI-compatible API
                    self.ai_clients[provider] = openai.OpenAI(base_url=config["api_base"])
                    logger.info(f"Initialized {provider} client")
                elif provider == "openrouter" and config["api_key"]:
                    # OpenRouter uses OpenAI-compatible API
                    self.ai_clients[provider] = openai.OpenAI(
                        api_key=config["api_key"],
                        base_url="https://openrouter.ai/api/v1"
                    )
                    logger.info(f"Initialized {provider} client")
                elif provider == "deepseek" and config["api_key"]:
                    # DeepSeek uses OpenAI-compatible API
                    self.ai_clients[provider] = openai.OpenAI(
                        api_key=config["api_key"],
                        base_url="https://api.deepseek.com/v1"
                    )
                    logger.info(f"Initialized {provider} client")
                elif provider == "google" and config["api_key"]:
                    # Import google.generativeai only if needed
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=config["api_key"])
                        self.ai_clients[provider] = genai
                        logger.info(f"Initialized {provider} client")
                    except ImportError:
                        logger.warning(f"Google Generative AI package not installed. To use Google models, install with: pip install google-generativeai")
                elif provider == "xai" and config["api_key"]:
                    # XAI client is not available in PyPI yet
                    # We'll use the OpenAI client with a custom base URL when it becomes available
                    logger.warning(f"XAI client not available. Using mock client.")
                    self.ai_clients[provider] = None
                elif provider == "qwen" and config["api_key"]:
                    # Qwen client is not available in PyPI yet
                    # We'll use the OpenAI client with a custom base URL when it becomes available
                    logger.warning(f"Qwen client not available. Using mock client.")
                    self.ai_clients[provider] = None
                elif provider == "microsoft" and config["api_key"] and config["endpoint"]:
                    # Import azure-openai only if needed
                    try:
                        from azure.openai import AzureOpenAI
                        self.ai_clients[provider] = AzureOpenAI(
                            api_key=config["api_key"],
                            api_version="2023-05-15",
                            azure_endpoint=config["endpoint"]
                        )
                        logger.info(f"Initialized {provider} client")
                    except ImportError:
                        # Try using the azure-ai-ml package instead
                        try:
                            # Use OpenAI client with Azure endpoint
                            self.ai_clients[provider] = openai.AzureOpenAI(
                                api_key=config["api_key"],
                                api_version="2023-05-15",
                                azure_endpoint=config["endpoint"]
                            )
                            logger.info(f"Initialized {provider} client using OpenAI Azure client")
                        except (ImportError, AttributeError):
                            logger.warning(f"Azure OpenAI package not installed. To use Microsoft models, install with: pip install azure-openai")
            except Exception as e:
                logger.error(f"Failed to initialize {provider} client: {str(e)}")
    
    def register_model(self, model_type: str, config: Dict[str, Any]):
        """
        Register a new model type with configuration.
        
        Args:
            model_type: The type/name of the model
            config: The model configuration
        """
        self.model_registry[model_type] = config
        logger.info(f"Registered model: {model_type}")
    
    async def route_task(
        self, 
        task: str, 
        model_type: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Route a task to a specific model type.
        
        Args:
            task: The task description
            model_type: The type of model to use
            options: Optional routing options
            
        Returns:
            The model's response
        """
        options = options or {}
        model_config = self.model_registry.get(model_type)
        if not model_config:
            raise ValueError(f"Unknown model type: {model_type}")
        
        provider = options.get("provider", model_config.get("provider", DEFAULT_PROVIDER))
        model = options.get("model", model_config.get("model"))
        
        context = await self.context_engine.get_model_context(task)
        additional_instructions = options.get("additional_instructions", "")
        full_prompt = f"{context}\n\nTASK:\n{task}\n{additional_instructions}"
        
        try:
            logger.info(f"Routing task to {model_type} model using {provider} provider and {model} model")
            
            # Check if provider client is available
            if provider not in self.ai_clients or self.ai_clients[provider] is None:
                logger.warning(f"{provider} client not available. Returning mock response.")
                return f"[MOCK RESPONSE] This is a simulated response for: {task}"
            
            client = self.ai_clients[provider]
            
            # Handle different provider APIs
            if provider == "openai" or provider in ["groq", "ollama", "lmstudio", "openrouter", "deepseek"]:
                # OpenAI-compatible API
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": model_config["system_prompt"]},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=model_config.get("temperature", 0.7)
                )
                return response.choices[0].message.content
            
            elif provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    system=model_config["system_prompt"],
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=model_config.get("temperature", 0.7)
                )
                return response.content[0].text
            
            elif provider == "mistral":
                response = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": model_config["system_prompt"]},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=model_config.get("temperature", 0.7)
                )
                return response.choices[0].message.content
            
            elif provider == "cohere":
                response = client.generate(
                    prompt=full_prompt,
                    model=model,
                    temperature=model_config.get("temperature", 0.7)
                )
                return response.generations[0].text
            
            elif provider == "google":
                genai = client
                model = genai.GenerativeModel(model_name=model)
                response = model.generate_content(
                    [model_config["system_prompt"], full_prompt],
                    generation_config=genai.GenerationConfig(
                        temperature=model_config.get("temperature", 0.7)
                    )
                )
                return response.text
            
            elif provider == "xai":
                # XAI client is not available yet
                # When it becomes available, we'll implement the API call
                logger.warning(f"XAI client not fully implemented. Returning mock response.")
                return f"[MOCK RESPONSE] This is a simulated response for XAI: {task}"
            
            elif provider == "qwen":
                # Qwen client is not available yet
                # When it becomes available, we'll implement the API call
                logger.warning(f"Qwen client not fully implemented. Returning mock response.")
                return f"[MOCK RESPONSE] This is a simulated response for Qwen: {task}"
            
            elif provider == "microsoft":
                # Try to use the Azure OpenAI client
                try:
                    response = client.chat.completions.create(
                        model=model.replace("azure-", ""),
                        messages=[
                            {"role": "system", "content": model_config["system_prompt"]},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=model_config.get("temperature", 0.7)
                    )
                    return response.choices[0].message.content
                except AttributeError:
                    # If the client doesn't have chat.completions, try a different approach
                    logger.warning(f"Azure OpenAI client doesn't support chat completions. Returning mock response.")
                    return f"[MOCK RESPONSE] This is a simulated response for Azure: {task}"
            
            else:
                logger.error(f"Unsupported provider: {provider}")
                return f"[ERROR] Unsupported provider: {provider}"
        
        except Exception as e:
            logger.error(f"Error routing task to {model_type} model: {str(e)}")
            return f"[ERROR] Failed to get response from {provider}: {str(e)}"
    
    async def route_multi_stage(
        self, 
        task: str, 
        stages: List[Dict[str, Any]], 
        options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Route a task through multiple stages of models.
        
        Args:
            task: The initial task description
            stages: List of stage configurations
            options: Optional routing options
            
        Returns:
            List of stage results
        """
        options = options or {}
        results = []
        current_task = task
        
        for i, stage in enumerate(stages):
            model_type = stage.get("model_type")
            if not model_type:
                raise ValueError(f"Missing model_type in stage {i}")
            
            stage_options = {**options, **stage.get("options", {})}
            
            # Add previous stage results to the context
            if i > 0 and stage.get("include_previous_results", True):
                previous_results = "\n\n".join([f"Stage {j+1} Result:\n{result}" for j, result in enumerate(results)])
                current_task = f"{current_task}\n\nPrevious Stage Results:\n{previous_results}"
            
            result = await self.route_task(current_task, model_type, stage_options)
            results.append(result)
            
            # Update the task for the next stage if specified
            if stage.get("update_task", False):
                current_task = result
        
        return results
