import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Type
from loguru import logger

from .state import ProcessorState
from .utils.config import ConfigManager
from .utils.error_handling import ErrorHandler, ErrorSeverity
from .utils.imports import ComponentLoader, import_optional
from .utils.caching import ResponseCache, generate_cache_key
from .metrics import MetricsCollector
from .optimizer import Optimizer
from .dispatcher import Dispatcher


class ProcessorEngine:
    """
    Core processor engine for multi-model processing.
    
    This class is responsible for:
    - Initializing and managing model handlers
    - Processing tasks with appropriate models
    - Integrating with the Latent Reasoning module
    - Error handling and recovery
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str]] = None,
        latent_reasoning_engine: Optional[Any] = None,
        context_engine: Optional[Any] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize the processor engine.
        
        Args:
            config: Configuration dictionary or path to config file
            latent_reasoning_engine: Optional latent reasoning engine
            context_engine: Optional context engine
            error_handler: Optional error handler
        """
        # Initialize configuration
        if isinstance(config, str):
            self.config_manager = ConfigManager(config_path=config)
        else:
            self.config_manager = ConfigManager(config=config)
            
        self.config = self.config_manager.get_all()
        
        # Initialize error handler
        self.error_handler = error_handler or ErrorHandler()
        
        # Initialize component loader
        self.component_loader = ComponentLoader()
        
        # Set up external engines
        self.latent_reasoning_engine = latent_reasoning_engine
        self.context_engine = context_engine
        
        # Initialize model handlers
        self.handlers: Dict[str, Any] = {}
        self._init_handlers()
        
        # Initialize metrics collector
        metrics_module = import_optional(".metrics")
        self.metrics = None
        if metrics_module:
            self.metrics = MetricsCollector(self.config.get("metrics", {}))
        
        # Initialize optimizer
        optimizer_module = import_optional(".optimizer")
        self.optimizer = None
        if optimizer_module and hasattr(optimizer_module, "Optimizer"):
            self.optimizer = optimizer_module.Optimizer(self.config.get("optimizer", {}))
        else:
            self.optimizer = Optimizer(self.config.get("optimizer", {}))
            
        # Initialize dispatcher
        dispatcher_module = import_optional(".dispatcher")
        self.dispatcher = None
        if dispatcher_module and hasattr(dispatcher_module, "Dispatcher"):
            self.dispatcher = dispatcher_module.Dispatcher(self.config.get("dispatcher", {}))
        else:
            self.dispatcher = Dispatcher(self.config.get("dispatcher", {}))
            
        # Initialize response cache if enabled
        self.cache = None
        if self.config.get("processor", {}).get("enable_caching", True):
            cache_config = self.config.get("cache", {})
            max_size = cache_config.get("max_size", 1000)
            default_ttl = cache_config.get("default_ttl", 3600)
            self.cache = ResponseCache(max_size=max_size, default_ttl=default_ttl)
        
        logger.info("ProcessorEngine initialized successfully")

    def _init_handlers(self) -> None:
        """Initialize model handlers."""
        models_config = self.config_manager.get("models", {})
        
        for model_id, model_config in models_config.items():
            handler_type = model_config.get("handler_type", "unknown")
            
            try:
                self._register_handler(model_id, handler_type, model_config)
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context={"model_id": model_id, "handler_type": handler_type},
                    severity=ErrorSeverity.WARNING
                )
                logger.warning(f"Failed to register handler for {model_id}: {str(e)}")

    def _register_handler(self, model_id: str, handler_type: str, config: Dict[str, Any]) -> None:
        """
        Register a model handler.
        
        Args:
            model_id: Identifier for the model
            handler_type: Type of handler to register
            config: Handler configuration
        """
        # Try to load the appropriate handler class
        handler_module_path = f".handlers.{handler_type}_handler"
        handler_class_name = f"{handler_type.capitalize()}ModelHandler"
        
        handler_module = import_optional(handler_module_path, "ptolemy.multi_model_processor")
        
        if handler_module and hasattr(handler_module, handler_class_name):
            handler_class = getattr(handler_module, handler_class_name)
            handler = handler_class(config)
            self.handlers[model_id] = handler
            logger.info(f"Registered {handler_type} handler for model {model_id}")
        else:
            logger.warning(f"Handler type {handler_type} not found for model {model_id}")
            # Fall back to mock handler if available
            self._register_mock_handler(model_id, config)

    def _register_mock_handler(self, model_id: str = "mock-model", config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a mock handler for testing or as fallback.
        
        Args:
            model_id: Identifier for the mock model
            config: Optional configuration for the mock handler
        """
        mock_module = import_optional(".handlers.mock_handler", "ptolemy.multi_model_processor")
        
        if mock_module and hasattr(mock_module, "MockModelHandler"):
            mock_handler = mock_module.MockModelHandler(config or {})
            self.handlers[model_id] = mock_handler
            logger.info(f"Registered mock handler as {model_id}")
        else:
            # If even mock handler can't be loaded, create minimal handler
            from .handlers.base_handler import BaseModelHandler
            
            class MinimalHandler(BaseModelHandler):
                async def process(self, prompt, parameters=None):
                    return {
                        "text": f"Minimal processed: {prompt[:20]}...",
                        "model": "minimal",
                        "tokens": {"prompt": len(prompt.split()), "completion": 10}
                    }
                    
                async def get_capabilities(self):
                    return {
                        "max_tokens": 100,
                        "supports_streaming": False,
                        "strengths": ["minimal", "fallback"]
                    }
            
            self.handlers[model_id] = MinimalHandler()
            logger.warning(f"Created minimal handler as {model_id}")

    async def process_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task with the multi-model processor.
        
        Args:
            task: The task to process
            context: Optional context dictionary
            model_preference: Optional preferred model
            
        Returns:
            Processing result
        """
        context = context or {}
        start_time = time.time()
        
        # Create processor state
        state = ProcessorState.create(task=task, context=context)
        
        # Create metrics for this task if metrics collection is enabled
        task_metrics = None
        if self.metrics:
            task_metrics = self.metrics.create_task_metrics(state.task_id)
        
        try:
            # Start processing pipeline
            state.advance_stage("analyzing")
            
            # Analyze task using optimizer if available
            analysis = {}
            if self.optimizer:
                analysis = await self.optimizer.analyze_task(task, context)
                
            # Retrieve relevant context if context engine is available
            auto_context = self.config.get("processor", {}).get("auto_context", True)
            if self.context_engine and auto_context:
                state.advance_stage("retrieving_context")
                try:
                    relevant_context = await self._retrieve_context(task, context)
                    if relevant_context:
                        context["retrieved_context"] = relevant_context
                except Exception as e:
                    self.error_handler.handle_error(
                        e, 
                        ErrorSeverity.WARNING,
                        "Error retrieving context, continuing without it",
                        {"task": task}
                    )
            
            # Apply latent reasoning if available and enabled
            latent_reasoning_result = None
            use_reasoning = analysis.get("use_reasoning", True)
            
            if self.latent_reasoning_engine and use_reasoning:
                state.advance_stage("reasoning")
                try:
                    iterations = analysis.get("reasoning_iterations", 
                                              self.config.get("latent_reasoning", {}).get("default_iterations", 3))
                    
                    latent_reasoning_result = await self._apply_latent_reasoning(
                        task, 
                        context,
                        iterations=iterations
                    )
                    
                    # Add reasoning result to context
                    if latent_reasoning_result:
                        context["reasoning_result"] = latent_reasoning_result
                except Exception as e:
                    self.error_handler.handle_error(
                        e, 
                        ErrorSeverity.WARNING,
                        "Error in latent reasoning, continuing without it",
                        {"task": task}
                    )
            
            # Select model
            state.advance_stage("selecting_model")
            
            # Use model preference if provided
            selected_model = model_preference
            
            # Otherwise, use optimizer to select model
            if not selected_model and self.optimizer:
                # Get available models
                available_models = await self.get_available_models()
                
                # Select using optimizer
                selected_model = await self.optimizer.select_optimal_model(
                    task, 
                    {"analysis": analysis, "context": context},
                    available_models
                )
            
            # Fall back to default if no model selected
            if not selected_model:
                selected_model = self.config.get("processor", {}).get("default_model", "mock-model")
                
            # Record selected model
            state.select_model(selected_model)
            
            # Prepare parameters for model
            parameters = {
                "model": selected_model,
                "context": context
            }
            
            # Add any task-specific parameters
            if "model_parameters" in context:
                parameters.update(context["model_parameters"])
                
            # Try to process with cache if available
            cached_response = None
            if self.cache and not context.get("skip_cache", False):
                cache_key = generate_cache_key(task, selected_model, parameters)
                cached_response = self.cache.get(cache_key)
                
            if cached_response:
                # Use cached response
                logger.info(f"Using cached response for task {state.task_id[:8]}...")
                response = cached_response
                state.advance_stage("completed")
            else:
                # Process with selected model
                state.advance_stage("processing")
                response = await self._process_with_model(task, selected_model, parameters, state)
                
                # Cache response if caching is enabled
                if self.cache and not context.get("skip_cache", False):
                    cache_key = generate_cache_key(task, selected_model, parameters)
                    self.cache.set(cache_key, response)
                
            # Extract and record token usage
            if "tokens" in response:
                state.record_token_usage(
                    model=selected_model,
                    prompt_tokens=response["tokens"].get("prompt", 0),
                    completion_tokens=response["tokens"].get("completion", 0)
                )
                
                # Record in metrics
                if task_metrics:
                    task_metrics.record_model_usage(
                        model_id=selected_model,
                        tokens=response["tokens"],
                        latency=response.get("metadata", {}).get("processing_time", 0)
                    )
            
            # Prepare result
            result = {
                "success": True,
                "output": response.get("text", ""),
                "model": selected_model,
                "tokens": response.get("tokens", {}),
                "time": time.time() - start_time,
                "task_id": state.task_id,
                "state": state.to_dict()
            }
            
            # Complete task metrics
            if task_metrics:
                task_metrics.complete()
                if self.metrics:
                    self.metrics.task_completed(state.task_id, success=True)
                
            return result
            
        except Exception as e:
            # Handle error and try to recover
            error_info = self.error_handler.handle_error(
                e, 
                ErrorSeverity.ERROR,
                "Error processing task",
                {"task": task, "task_id": state.task_id}
            )
            
            # Record error in state
            state.record_error(error_info)
            
            # Record in metrics
            if task_metrics:
                task_metrics.record_error(error_info)
                if self.metrics:
                    self.metrics.task_completed(state.task_id, success=False)
            
            # Return error result
            return {
                "success": False,
                "error": str(e),
                "task_id": state.task_id,
                "time": time.time() - start_time,
                "state": state.to_dict()
            }

    async def process_task_stream(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None, 
        model_preference: Optional[str] = None, 
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Process a task with streaming response.
        
        Args:
            task: The task description or query
            context: Optional context information
            model_preference: Optional preferred model to use
            options: Optional processing options
            
        Yields:
            Response chunks as they become available
        """
        context = context or {}
        options = options or {}
        options["streaming"] = True
        
        # Create state and start processing
        task_id = str(uuid.uuid4())
        state = ProcessorState.create(task=task, context=context)
        
        try:
            # Analyze task and get parameters
            state.advance_stage("optimization")
            parameters = {}
            if self.optimizer:
                parameters = await self.optimizer.analyze_task(task, context)
            else:
                parameters = {"use_reasoning": True, "reasoning_iterations": 3}
            
            # Override with options
            parameters.update({k: v for k, v in options.items() if v is not None})
            
            # Integrate with latent reasoning if available
            if self.latent_reasoning_engine and parameters.get("use_reasoning", True):
                state.advance_stage("latent_reasoning")
                reasoning_result = await self._apply_latent_reasoning(
                    task, 
                    context,
                    iterations=parameters.get("reasoning_iterations", 3)
                )
                
                if reasoning_result:
                    state.reasoning_output = reasoning_result.get("output")
                    state.reasoning_iterations = reasoning_result.get("iterations", 0)
                    context["reasoning_result"] = reasoning_result
                
                # Add reasoning info to the first chunk
                yield {
                    "event": "reasoning_complete",
                    "iterations": state.reasoning_iterations,
                    "task_id": task_id
                }
            
            # Select model
            state.advance_stage("model_selection")
            model = model_preference
            
            if not model and self.optimizer:
                model = await self.optimizer.select_optimal_model(
                    task, parameters, available_models=list(self.handlers.keys())
                )
            elif not model and self.dispatcher:
                model = self.dispatcher.get_preferred_model(parameters.get("task_type", "general"))
                
            if not model and self.handlers:
                # Default to first registered handler
                model = next(iter(self.handlers.keys()))
                
            state.selected_model = model
            
            # Check cache if enabled
            if self.cache and not parameters.get("skip_cache", False):
                cache_key = generate_cache_key(task, model, context)
                cache_result = self.cache.get(cache_key)
                
                if cache_result:
                    # Cache hit - yield the cached result as a single chunk
                    state.advance_stage("cache_hit")
                    state.cache_hit = True
                    
                    yield {
                        "event": "cache_hit",
                        "text": cache_result.get("text", ""),
                        "finished": True,
                        "model": model,
                        "state": state.to_dict(),
                        "tokens": cache_result.get("tokens", {})
                    }
                    return
            
            # Prepare prompt
            state.advance_stage("prompt_preparation")
            prompt = task
            if self.optimizer:
                prompt = await self.optimizer.optimize_prompt(
                    task, context, model, getattr(state, "reasoning_output", None)
                )
            else:
                # Simple prompt preparation
                prompt_parts = []
                if context:
                    context_str = "\n".join([f"{k}: {v}" for k, v in context.items() 
                                            if k != "reasoning_result" and 
                                            not isinstance(v, dict) and 
                                            not isinstance(v, list)])
                    prompt_parts.append(f"Context:\n{context_str}")
                
                reasoning_output = None
                if "reasoning_result" in context:
                    reasoning_output = context["reasoning_result"].get("output")
                    if reasoning_output:
                        prompt_parts.append(f"Reasoning:\n{reasoning_output}")
                
                prompt_parts.append(f"Task: {task}")
                prompt = "\n\n".join(prompt_parts)
                
            # Dispatch to handler
            state.advance_stage("processing")
            
            if not self.dispatcher:
                # No dispatcher available, create a fallback
                from .dispatcher import Dispatcher
                self.dispatcher = Dispatcher(self.handlers)
                
            dispatch_result = await self.dispatcher.dispatch(task, model, prompt, parameters)
            
            # Check if we received a stream handler
            if "stream_handler" in dispatch_result:
                handler = dispatch_result["stream_handler"]
                
                # Start streaming
                token_count = {"prompt": 0, "completion": 0}
                async for chunk in handler.process_stream(prompt, parameters):
                    # Update token count if available
                    if "tokens" in chunk:
                        token_count["prompt"] += chunk["tokens"].get("prompt", 0)
                        token_count["completion"] += chunk["tokens"].get("completion", 0)
                    
                    # Add state information to final chunk if finished
                    if chunk.get("finished", False):
                        state.advance_stage("completed")
                        
                        # Record token usage
                        state.record_token_usage(
                            "processing", 
                            token_count["prompt"],
                            token_count["completion"]
                        )
                        
                        # Include state in final chunk
                        chunk["state"] = state.to_dict()
                        
                        # Cache the full result if caching is enabled
                        if self.cache and not parameters.get("skip_cache", False):
                            # Reconstruct the full response from all chunks
                            cache_data = {
                                "text": chunk.get("text", ""),
                                "model": model,
                                "tokens": token_count,
                                "timestamp": time.time()
                            }
                            cache_key = generate_cache_key(task, model, context)
                            self.cache.set(cache_key, cache_data)
                    
                    # Yield each chunk
                    yield chunk
            else:
                # Not a streaming handler, yield the entire result as one chunk
                yield {
                    "text": dispatch_result.get("text", ""),
                    "finished": True,
                    "model": model,
                    "state": state.to_dict(),
                    "tokens": dispatch_result.get("tokens", {})
                }
                
        except Exception as e:
            # Handle error
            state.advance_stage("error", {"error": str(e)})
            logger.error(f"Error in streaming process: {str(e)}")
            
            # Yield error response
            yield {
                "error": str(e),
                "finished": True,
                "model": state.selected_model or "unknown",
                "state": state.to_dict()
            }

    async def _retrieve_context(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant context for the task.
        
        Args:
            task: The task to process
            context: The current context
            
        Returns:
            Retrieved context
        """
        # Retrieve context using context engine
        return await self.context_engine.retrieve_context(task, context)

    async def _apply_latent_reasoning(
        self, 
        task: str, 
        context: Dict[str, Any], 
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Apply latent reasoning to the task.
        
        Args:
            task: The task to process
            context: The current context
            iterations: Number of reasoning iterations
            
        Returns:
            Reasoning result
        """
        # Apply latent reasoning using latent reasoning engine
        return await self.latent_reasoning_engine.process(task, context, iterations)

    async def _process_with_model(
        self, 
        task: str, 
        model: str, 
        parameters: Dict[str, Any], 
        state: ProcessorState
    ) -> Dict[str, Any]:
        """
        Process the task with the selected model.
        
        Args:
            task: The task to process
            model: The selected model
            parameters: Processing parameters
            state: The processor state
            
        Returns:
            Processing result
        """
        # Process with the selected model using dispatcher
        return await self.dispatcher.dispatch(task, model, parameters, state)

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models with their capabilities.
        
        Returns:
            List of available models with capabilities
        """
        models = []
        
        for model_id, handler in self.handlers.items():
            try:
                capabilities = await handler.get_capabilities()
                models.append({
                    "id": model_id,
                    "capabilities": capabilities
                })
            except Exception as e:
                logger.warning(f"Error getting capabilities for {model_id}: {str(e)}")
                
        return models

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current processing metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.get_metrics() if hasattr(self.metrics, "get_metrics") else {}