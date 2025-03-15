"""
Model Adapter Module

This module provides utilities for adapting prompts and parameters
between different model providers, ensuring cross-model compatibility.
"""

from typing import Dict, Any, List, Optional, Callable


class ModelAdapter:
    """
    Adapts prompts and parameters between different model providers.
    
    This utility helps ensure consistent behavior when switching between
    different model providers by adapting prompts and parameters to the
    expected format for each provider.
    """
    
    def __init__(self):
        """Initialize the model adapter with format-specific adapters."""
        self.adapters = {
            "openai": self._adapt_openai(),
            "anthropic": self._adapt_anthropic(),
            "cohere": self._adapt_cohere(),
            "local": self._adapt_local()
        }
    
    def adapt_prompt(self, prompt: str, source_format: str, target_format: str) -> str:
        """
        Adapt a prompt from one model format to another.
        
        Args:
            prompt: Original prompt
            source_format: Source model format (e.g., 'openai')
            target_format: Target model format (e.g., 'anthropic')
            
        Returns:
            Adapted prompt
        """
        # If formats are the same, no adaptation needed
        if source_format == target_format:
            return prompt
            
        # Check if we have adapters for both formats
        if source_format not in self.adapters or target_format not in self.adapters:
            return prompt  # Return original if we can't adapt
            
        # Parse the prompt using source format
        parsed = self.adapters[source_format]["parse"](prompt)
        
        # Generate prompt in target format
        return self.adapters[target_format]["generate"](parsed)
    
    def adapt_parameters(
        self, 
        parameters: Dict[str, Any], 
        source_format: str, 
        target_format: str
    ) -> Dict[str, Any]:
        """
        Adapt parameters from one model format to another.
        
        Args:
            parameters: Original parameters
            source_format: Source model format
            target_format: Target model format
            
        Returns:
            Adapted parameters
        """
        # If formats are the same, no adaptation needed
        if source_format == target_format:
            return parameters.copy()
            
        # Check if we have adapters for both formats
        if source_format not in self.adapters or target_format not in self.adapters:
            return parameters.copy()  # Return original if we can't adapt
            
        # Map parameters using the parameter mapping
        param_mapping = self._get_parameter_mapping(source_format, target_format)
        
        adapted = {}
        for src_param, target_param in param_mapping.items():
            if src_param in parameters:
                adapted[target_param] = parameters[src_param]
                
        return adapted
    
    def _get_parameter_mapping(self, source: str, target: str) -> Dict[str, str]:
        """Get parameter mapping between model formats."""
        # Common parameter mappings
        mappings = {
            "openai_to_anthropic": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
                "stop": "stop_sequences"
            },
            "anthropic_to_openai": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "top_k": "top_k",
                "stop_sequences": "stop"
            },
            "openai_to_cohere": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty"
            },
            "cohere_to_openai": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty"
            },
            "anthropic_to_cohere": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "p",
                "stop_sequences": "stop_sequences"
            },
            "cohere_to_anthropic": {
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "p": "top_p",
                "stop_sequences": "stop_sequences"
            }
        }
        
        # Generate key for mapping lookup
        mapping_key = f"{source}_to_{target}"
        
        # Return appropriate mapping or empty dict if not found
        return mappings.get(mapping_key, {})
    
    # Format-specific adapters
    def _adapt_openai(self) -> Dict[str, Callable]:
        """Get the OpenAI format adapter."""
        return {
            "parse": self._parse_openai_prompt,
            "generate": self._generate_openai_prompt
        }
    
    def _adapt_anthropic(self) -> Dict[str, Callable]:
        """Get the Anthropic format adapter."""
        return {
            "parse": self._parse_anthropic_prompt,
            "generate": self._generate_anthropic_prompt
        }
    
    def _adapt_cohere(self) -> Dict[str, Callable]:
        """Get the Cohere format adapter."""
        return {
            "parse": self._parse_cohere_prompt,
            "generate": self._generate_cohere_prompt
        }
    
    def _adapt_local(self) -> Dict[str, Callable]:
        """Get the local format adapter."""
        return {
            "parse": self._parse_local_prompt,
            "generate": self._generate_local_prompt
        }
    
    # Parsing and generation methods
    def _parse_openai_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse an OpenAI-formatted prompt."""
        # Check if prompt contains message format indicators
        if "system:" in prompt.lower() or "user:" in prompt.lower() or "assistant:" in prompt.lower():
            messages = []
            current_role = None
            current_content = []
            
            for line in prompt.split('\n'):
                line = line.strip()
                
                # Check for role markers
                if line.lower().startswith("system:"):
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "system"
                    current_content = [line[7:].strip()]
                elif line.lower().startswith("user:"):
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "user"
                    current_content = [line[5:].strip()]
                elif line.lower().startswith("assistant:"):
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "assistant"
                    current_content = [line[10:].strip()]
                else:
                    current_content.append(line)
            
            # Add the last message if there is one
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                
            return {"messages": messages, "format": "openai_chat"}
        else:
            # Simple prompt format
            return {"content": prompt, "format": "openai_completion"}
    
    def _generate_openai_prompt(self, parsed: Dict[str, Any]) -> str:
        """Generate an OpenAI-formatted prompt."""
        if parsed.get("format") == "openai_chat" and "messages" in parsed:
            # Convert messages back to string format
            result = []
            for msg in parsed["messages"]:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                result.append(f"{role}: {content}")
            return "\n\n".join(result)
        else:
            # Simple content
            return parsed.get("content", "")
    
    def _parse_anthropic_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse an Anthropic-formatted prompt."""
        # Check for Anthropic's format markers
        if "human:" in prompt.lower() and "assistant:" in prompt.lower():
            # Parse Claude's Human/Assistant format
            parts = {"system": "", "human": "", "assistant": ""}
            current_part = None
            current_content = []
            
            for line in prompt.split('\n'):
                line = line.strip()
                
                if line.lower().startswith("human:"):
                    if current_part:
                        parts[current_part] = "\n".join(current_content).strip()
                    current_part = "human"
                    current_content = [line[6:].strip()]
                elif line.lower().startswith("assistant:"):
                    if current_part:
                        parts[current_part] = "\n".join(current_content).strip()
                    current_part = "assistant"
                    current_content = [line[10:].strip()]
                else:
                    if current_part:
                        current_content.append(line)
            
            # Add the last part if there is one
            if current_part:
                parts[current_part] = "\n".join(current_content).strip()
                
            return {
                "human": parts["human"],
                "assistant": parts["assistant"],
                "format": "anthropic"
            }
        else:
            # Assume it's just the human part
            return {"human": prompt, "assistant": "", "format": "anthropic"}
    
    def _generate_anthropic_prompt(self, parsed: Dict[str, Any]) -> str:
        """Generate an Anthropic-formatted prompt."""
        if parsed.get("format") == "openai_chat" and "messages" in parsed:
            # Convert OpenAI chat messages to Anthropic format
            human_parts = []
            assistant_parts = []
            
            for msg in parsed["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    human_parts.append(content)
                elif role == "assistant":
                    assistant_parts.append(content)
                elif role == "system":
                    # Prepend system message to human message
                    human_parts.insert(0, f"[System Instruction]\n{content}\n[/System Instruction]")
            
            # Combine parts
            human = "\n\n".join(human_parts)
            assistant = "\n\n".join(assistant_parts)
            
            if assistant:
                return f"Human: {human}\n\nAssistant: {assistant}"
            else:
                return f"Human: {human}\n\nAssistant: "
        else:
            # Simple human/assistant format
            human = parsed.get("human", parsed.get("content", ""))
            assistant = parsed.get("assistant", "")
            
            if assistant:
                return f"Human: {human}\n\nAssistant: {assistant}"
            else:
                return f"Human: {human}\n\nAssistant: "
    
    def _parse_cohere_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a Cohere-formatted prompt."""
        # Cohere doesn't have a specific format, so just return the content
        return {"content": prompt, "format": "cohere"}
    
    def _generate_cohere_prompt(self, parsed: Dict[str, Any]) -> str:
        """Generate a Cohere-formatted prompt."""
        if parsed.get("format") == "openai_chat" and "messages" in parsed:
            # Convert OpenAI chat to Cohere format (concatenated)
            result = []
            for msg in parsed["messages"]:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                result.append(f"{role}: {content}")
            return "\n\n".join(result)
        elif parsed.get("format") == "anthropic":
            # Convert Anthropic format to Cohere
            human = parsed.get("human", "")
            assistant = parsed.get("assistant", "")
            result = [f"Human: {human}"]
            if assistant:
                result.append(f"Assistant: {assistant}")
            return "\n\n".join(result)
        else:
            # Just return the content
            return parsed.get("content", "")
    
    def _parse_local_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a local model formatted prompt."""
        # For local models, just return the content
        return {"content": prompt, "format": "local"}
    
    def _generate_local_prompt(self, parsed: Dict[str, Any]) -> str:
        """Generate a local model formatted prompt."""
        # Extract content based on the format
        if parsed.get("format") == "openai_chat" and "messages" in parsed:
            # Convert messages to a simple format for local models
            result = []
            for msg in parsed["messages"]:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                result.append(f"{role}: {content}")
            return "\n\n".join(result)
        elif parsed.get("format") == "anthropic":
            human = parsed.get("human", "")
            assistant = parsed.get("assistant", "")
            result = []
            if human:
                result.append(f"Human: {human}")
            if assistant:
                result.append(f"Assistant: {assistant}")
            return "\n\n".join(result)
        else:
            # Just return the content
            return parsed.get("content", "")
