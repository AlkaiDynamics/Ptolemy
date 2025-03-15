import asyncio
import random
import time
from typing import Dict, Any, Optional, List
from loguru import logger

from .base_handler import BaseModelHandler

class MockModelHandler(BaseModelHandler):
    """
    Mock model handler for testing the Multi-Model Processor.
    
    This handler simulates responses from a language model without requiring
    actual API calls, making it useful for testing and development.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock model handler.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.config = config or {}
        self.latency = self.config.get("latency", 0.5)  # Simulated latency in seconds
        self.error_rate = self.config.get("error_rate", 0.0)  # Probability of simulated failure
        self.models = self.config.get("models", ["mock-gpt", "mock-llama", "mock-claude"])
        self.default_model = self.config.get("default_model", self.models[0] if self.models else "mock-gpt")
        
        # Predefined responses for testing specific scenarios
        self.predefined_responses = self.config.get("predefined_responses", {})
        
        logger.debug(f"Initialized MockModelHandler with models: {self.models}")
    
    async def process(self, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a prompt with a simulated response.
        
        Args:
            prompt: The prompt to process
            parameters: Optional processing parameters
            
        Returns:
            Dictionary containing the generated response
        """
        parameters = parameters or {}
        start_time = time.time()
        
        # Check for predefined response based on prompt keywords
        response = self._get_predefined_response(prompt, parameters)
        if response:
            logger.debug("Using predefined response")
            self._simulate_latency()
            return response
        
        # Simulate processing delay
        self._simulate_latency(parameters.get("speed", None))
        
        # Simulate random errors
        if random.random() < self.error_rate:
            error_msg = "Simulated random processing error"
            logger.warning(error_msg)
            raise RuntimeError(error_msg)
        
        # Generate mock response
        model_id = parameters.get("model_id", self.default_model)
        input_tokens = await self.estimate_tokens(prompt)
        
        # Generate response length based on prompt length
        response_length = min(
            len(prompt.split()) // 2 + random.randint(10, 30),
            parameters.get("max_tokens", 100)
        )
        
        # Create a simple response
        if "minimal_mode" in parameters and parameters["minimal_mode"]:
            response_text = f"Minimal processed response for: {prompt[:30]}..."
        else:
            response_parts = []
            # Add a basic mock response
            response_parts.append(f"This is a mock response from {model_id}.")
            
            # Reflect some content from the prompt
            if len(prompt) > 20:
                response_parts.append(f"You asked about: {prompt[:20]}...")
            
            # Add some simulated analysis
            response_parts.append("Here is my analysis:")
            response_parts.append("1. The task appears to be about " + self._generate_mock_subject(prompt))
            response_parts.append("2. Key points to consider include " + self._generate_mock_points())
            response_parts.append("3. " + self._generate_mock_conclusion())
            
            response_text = "\n".join(response_parts)
        
        # Calculate token metrics
        output_tokens = await self.estimate_tokens(response_text)
        
        # Record metrics
        model_id = parameters.get("model_id", self.default_model)
        self.record_metrics(start_time, input_tokens, output_tokens, model_id)
        
        return {
            "text": response_text,
            "model": model_id,
            "tokens": {
                "prompt": input_tokens,
                "completion": output_tokens
            },
            "metadata": {
                "mock": True,
                "processing_time": time.time() - start_time,
                "simulated_complexity": self._calculate_mock_complexity(prompt)
            }
        }
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this mock model.
        
        Returns:
            Dictionary of model capabilities
        """
        return {
            "max_tokens": self.config.get("max_tokens", 2048),
            "supports_streaming": self.config.get("supports_streaming", True),
            "strengths": [
                "testing",
                "development",
                "debugging",
                "mockery"
            ],
            "weaknesses": [
                "not a real model",
                "provides simulated responses only",
                "limited variety"
            ],
            "models": self.models
        }
    
    def _get_predefined_response(self, prompt: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if there's a predefined response for this prompt.
        
        Args:
            prompt: The input prompt
            parameters: Processing parameters
            
        Returns:
            Predefined response or None
        """
        # Check if we have specific keyword matches
        for keyword, response_data in self.predefined_responses.items():
            if keyword.lower() in prompt.lower():
                model_id = parameters.get("model_id", self.default_model)
                
                # If the response is a dict with text, use as is
                if isinstance(response_data, dict) and "text" in response_data:
                    response = response_data.copy()
                    if "model" not in response:
                        response["model"] = model_id
                    return response
                
                # If it's a string, build a proper response dict
                if isinstance(response_data, str):
                    input_tokens = len(prompt.split())
                    output_tokens = len(response_data.split())
                    
                    return {
                        "text": response_data,
                        "model": model_id,
                        "tokens": {
                            "prompt": input_tokens,
                            "completion": output_tokens
                        },
                        "metadata": {
                            "mock": True,
                            "predefined": True
                        }
                    }
        
        return None
    
    def _simulate_latency(self, speed: Optional[str] = None) -> None:
        """
        Simulate processing latency.
        
        Args:
            speed: Optional speed setting (fast, slow)
        """
        latency = self.latency
        
        if speed == "fast":
            latency = latency * 0.5
        elif speed == "slow":
            latency = latency * 2.0
            
        # Add some randomness to latency
        latency = latency * (0.8 + random.random() * 0.4)
        
        time.sleep(latency)
    
    def _calculate_mock_complexity(self, prompt: str) -> str:
        """
        Calculate a mock complexity score for a prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Complexity level string
        """
        length = len(prompt)
        words = len(prompt.split())
        
        if length > 500 or words > 100:
            return "high"
        elif length > 200 or words > 50:
            return "medium"
        else:
            return "low"
    
    def _generate_mock_subject(self, prompt: str) -> str:
        """
        Generate a mock subject based on the prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Mock subject string
        """
        subjects = [
            "data processing", 
            "system architecture", 
            "machine learning",
            "natural language understanding", 
            "software development",
            "optimization strategies",
            "knowledge management"
        ]
        
        # If prompt contains certain keywords, use them to select a relevant subject
        if "data" in prompt.lower():
            return "data processing and analysis"
        elif any(term in prompt.lower() for term in ["model", "ml", "ai"]):
            return "artificial intelligence and machine learning"
        elif any(term in prompt.lower() for term in ["code", "programming", "software"]):
            return "software development and system architecture"
        
        # Otherwise select random subject
        return random.choice(subjects)
    
    def _generate_mock_points(self) -> str:
        """
        Generate mock discussion points.
        
        Returns:
            String with mock points
        """
        points = [
            "efficiency, scalability, and maintainability",
            "performance optimization, data structure selection, and algorithm complexity",
            "modular design, error handling, and documentation",
            "integration testing, edge cases, and reliability",
            "resource management, concurrency, and parallelism"
        ]
        return random.choice(points)
    
    def _generate_mock_conclusion(self) -> str:
        """
        Generate a mock conclusion.
        
        Returns:
            String with mock conclusion
        """
        conclusions = [
            "Overall, this approach offers a good balance of performance and simplicity.",
            "The implementation should focus on maintainability and future extensibility.",
            "Consider trade-offs between performance optimization and code readability.",
            "A modular architecture would allow for more flexibility in the long term.",
            "Proper error handling and logging will be essential for production use."
        ]
        return random.choice(conclusions)
