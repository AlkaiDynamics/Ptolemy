import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from loguru import logger

from .base_handler import BaseModelHandler
from ..utils.imports import import_optional

# Import OpenAI library with fallback
openai = import_optional("openai")
tiktoken = import_optional("tiktoken")


class OpenAIModelHandler(BaseModelHandler):
    """
    Handler for OpenAI API models (GPT-3.5, GPT-4, etc.).
    
    This handler implements the BaseModelHandler interface for OpenAI's models,
    providing a consistent way to process prompts through various OpenAI models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI model handler.
        
        Args:
            config: Configuration dictionary with OpenAI-specific settings
        """
        super().__init__(config)
        self.config = config or {}
        
        # Check for OpenAI library
        if openai is None:
            raise ImportError(
                "OpenAI library not found. Install with 'pip install openai'"
            )
            
        # Initialize OpenAI client
        api_key = self.config.get("api_key")
        if not api_key:
            # Try to get from environment
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in config or OPENAI_API_KEY environment variable."
            )
            
        # Create client
        self.client = openai.OpenAI(api_key=api_key)
        
        # Default model configuration
        self.default_model = self.config.get("default_model", "gpt-3.5-turbo")
        self.model_configs = self.config.get("models", {})
        
        # Tiktoken encoding for token estimation
        self.encoding = None
        if tiktoken:
            try:
                self.encoding = tiktoken.encoding_for_model(self.default_model)
            except Exception as e:
                logger.warning(f"Could not load tiktoken encoding: {str(e)}")
                
        logger.info(f"Initialized OpenAI handler with default model: {self.default_model}")
        
    async def process(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a prompt with an OpenAI model.
        
        Args:
            prompt: The prompt to process
            parameters: Optional processing parameters
            
        Returns:
            Dictionary containing the response
        """
        parameters = parameters or {}
        start_time = time.time()
        
        # Prepare request parameters
        model = parameters.get("model", self.default_model)
        max_tokens = parameters.get("max_tokens", self.config.get("max_tokens", 1000))
        temperature = parameters.get("temperature", self.config.get("temperature", 0.7))
        top_p = parameters.get("top_p", self.config.get("top_p", 1.0))
        
        # Check if we should use messages format (ChatCompletion) or text format (Completion)
        use_chat_format = "gpt-" in model or parameters.get("use_chat_format", True)
        
        try:
            if use_chat_format:
                # Prepare messages from prompt
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                elif isinstance(prompt, list):
                    messages = prompt
                else:
                    raise ValueError(f"Invalid prompt format: {type(prompt)}")
                    
                # Add system message if provided
                system_prompt = parameters.get("system_prompt")
                if system_prompt and isinstance(messages, list):
                    messages.insert(0, {"role": "system", "content": system_prompt})
                
                # Estimate input tokens
                input_tokens = await self.estimate_tokens(json.dumps(messages))
                
                # Send request
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=parameters.get("n", 1),
                    stop=parameters.get("stop", None),
                    presence_penalty=parameters.get("presence_penalty", 0),
                    frequency_penalty=parameters.get("frequency_penalty", 0),
                    stream=False
                )
                
                # Extract response data
                response_text = response.choices[0].message.content
                output_tokens = response.usage.completion_tokens
                input_tokens = response.usage.prompt_tokens
                
            else:
                # Legacy completion API
                # Estimate input tokens
                input_tokens = await self.estimate_tokens(prompt)
                
                # Send request
                response = self.client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=parameters.get("n", 1),
                    stop=parameters.get("stop", None),
                    presence_penalty=parameters.get("presence_penalty", 0),
                    frequency_penalty=parameters.get("frequency_penalty", 0),
                    stream=False
                )
                
                # Extract response data
                response_text = response.choices[0].text
                output_tokens = response.usage.completion_tokens
                input_tokens = response.usage.prompt_tokens
            
            # Record metrics
            self.record_metrics(start_time, input_tokens, output_tokens, model)
            
            return {
                "text": response_text,
                "model": model,
                "tokens": {
                    "prompt": input_tokens,
                    "completion": output_tokens
                },
                "metadata": {
                    "finish_reason": response.choices[0].finish_reason,
                    "processing_time": time.time() - start_time,
                    "model_provider": "openai"
                }
            }
            
        except Exception as e:
            self.record_error(e, model)
            logger.error(f"Error in OpenAI API call: {str(e)}")
            raise
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this model.
        
        Returns:
            Dictionary of model capabilities
        """
        model_capabilities = {
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "supports_streaming": True,
                "strengths": ["general knowledge", "instruction following", "conversational"],
                "weaknesses": ["specialized knowledge", "large context", "reasoning"]
            },
            "gpt-4": {
                "max_tokens": 8192,
                "supports_streaming": True,
                "strengths": ["reasoning", "instruction following", "complex tasks", "nuance"],
                "weaknesses": ["cost", "speed", "specialized domain knowledge"]
            },
            "gpt-4-turbo": {
                "max_tokens": 128000,
                "supports_streaming": True,
                "strengths": ["reasoning", "instruction following", "complex tasks", "large context"],
                "weaknesses": ["cost", "specialized domain knowledge"]
            }
        }
        
        # Get default model capabilities or provided model if specified
        model = self.config.get("model", self.default_model)
        capabilities = model_capabilities.get(model, model_capabilities.get("gpt-3.5-turbo", {}))
        
        # Include available models
        capabilities["available_models"] = list(model_capabilities.keys())
        
        return capabilities
    
    async def estimate_tokens(self, text: Union[str, List, Dict]) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: The text to estimate or a message object/list
            
        Returns:
            Estimated token count
        """
        # If tiktoken is available, use it for accurate estimation
        if self.encoding is not None:
            try:
                if isinstance(text, str):
                    return len(self.encoding.encode(text))
                elif isinstance(text, (list, dict)):
                    # Convert to string first
                    json_text = json.dumps(text)
                    return len(self.encoding.encode(json_text))
            except Exception as e:
                logger.warning(f"Error estimating tokens with tiktoken: {str(e)}")
        
        # Fall back to simple estimation
        return super().estimate_tokens(str(text))
    
    async def preprocess(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Preprocess a prompt for this model.
        
        Args:
            prompt: The prompt to preprocess
            parameters: Optional preprocessing parameters
            
        Returns:
            Preprocessed prompt
        """
        parameters = parameters or {}
        
        # If system prompt is provided and the prompt is not already in message format,
        # convert it to messages format
        if (
            "system_prompt" in parameters and 
            isinstance(prompt, str) and 
            not isinstance(prompt, list)
        ):
            return [
                {"role": "system", "content": parameters["system_prompt"]},
                {"role": "user", "content": prompt}
            ]
            
        return prompt
