from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import asyncio
from loguru import logger

class BaseModelHandler(ABC):
    """
    Abstract base class for model handlers in the Multi-Model Processor.
    
    This class defines the interface that all model handlers must implement,
    providing a consistent way to interact with different language models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model handler.
        
        Args:
            config: Configuration dictionary for the handler
        """
        self.config = config or {}
        self.metrics: Dict[str, Any] = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_processing_time": 0.0,
            "average_response_time": 0.0,
            "calls_per_model": {},
            "errors": 0
        }
        
    @abstractmethod
    async def process(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a prompt with this model.
        
        Args:
            prompt: The prompt to process
            parameters: Optional processing parameters
            
        Returns:
            Dictionary containing:
            - text: The generated text
            - model: Identifier of the model used
            - tokens: Token usage information
            - metadata: Additional model-specific information
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this model.
        
        Returns:
            Dictionary of model capabilities including:
            - max_tokens: Maximum tokens supported
            - supports_streaming: Whether streaming is supported
            - strengths: List of model strengths
            - weaknesses: List of model weaknesses
        """
        pass
    
    async def preprocess(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Preprocess a prompt for this model.
        
        Args:
            prompt: The prompt to preprocess
            parameters: Optional preprocessing parameters
            
        Returns:
            Preprocessed prompt
        """
        # Default implementation - subclasses can override
        return prompt
        
    async def postprocess(self, response: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Postprocess a response from this model.
        
        Args:
            response: The raw response from the model
            parameters: Optional postprocessing parameters
            
        Returns:
            Processed response
        """
        # Default implementation - subclasses can override
        return response
    
    async def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: The text to estimate
            
        Returns:
            Estimated token count
        """
        # Simple estimation based on words and punctuation
        # This is a rough estimate - model-specific handlers should override with more accurate methods
        words = text.split()
        # Add extra tokens for punctuation and special characters
        extra_tokens = len([c for c in text if c in '.,;:!?()[]{}"\''])
        return len(words) + extra_tokens
        
    def record_metrics(self, start_time: float, input_tokens: int, output_tokens: int, model_id: str) -> None:
        """
        Record metrics for a model call.
        
        Args:
            start_time: Start time of the call
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_id: Identifier of the specific model used
        """
        processing_time = time.time() - start_time
        
        # Update total metrics
        self.metrics["total_calls"] += 1
        self.metrics["total_input_tokens"] += input_tokens
        self.metrics["total_output_tokens"] += output_tokens
        self.metrics["total_tokens"] += (input_tokens + output_tokens)
        self.metrics["total_processing_time"] += processing_time
        
        # Update average response time
        self.metrics["average_response_time"] = (
            self.metrics["total_processing_time"] / self.metrics["total_calls"]
        )
        
        # Update per-model metrics
        if model_id not in self.metrics["calls_per_model"]:
            self.metrics["calls_per_model"][model_id] = {
                "calls": 0,
                "tokens": 0,
                "processing_time": 0.0
            }
            
        model_metrics = self.metrics["calls_per_model"][model_id]
        model_metrics["calls"] += 1
        model_metrics["tokens"] += (input_tokens + output_tokens)
        model_metrics["processing_time"] += processing_time
        
        # Log metrics at debug level
        logger.debug(
            f"Model call: {model_id}, "
            f"Time: {processing_time:.2f}s, "
            f"Tokens: {input_tokens + output_tokens} "
            f"(in: {input_tokens}, out: {output_tokens})"
        )
        
    def record_error(self, error: Exception, model_id: Optional[str] = None) -> None:
        """
        Record an error that occurred during processing.
        
        Args:
            error: The exception that occurred
            model_id: Optional identifier of the specific model
        """
        self.metrics["errors"] += 1
        
        if model_id and model_id in self.metrics["calls_per_model"]:
            if "errors" not in self.metrics["calls_per_model"][model_id]:
                self.metrics["calls_per_model"][model_id]["errors"] = 0
            self.metrics["calls_per_model"][model_id]["errors"] += 1
            
        logger.error(f"Error in model handler: {str(error)}")
        
    async def check_availability(self) -> bool:
        """
        Check if the model is available for use.
        
        Returns:
            True if the model is available, False otherwise
        """
        try:
            capabilities = await self.get_capabilities()
            return True
        except Exception as e:
            logger.warning(f"Model unavailable: {str(e)}")
            return False
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics for this handler.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
