
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Type
from loguru import logger

from .state import ProcessorState
from .utils.config import ConfigManager
from .utils.error_handling import ErrorHandler, ErrorSeverity
from .utils.imports import ComponentLoader, import_optional


class ProcessorEngine:
    """
    Central orchestrator for the Multi-Model Processor.
    
    This class manages the processing of tasks across different language models,
    with support for routing, optimization, and integration with the Latent
    Reasoning Module.
    """
    
    def __init__(
        self, 
        latent_reasoning_engine=None,
        context_engine=None,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the processor engine.
        
        Args:
            latent_reasoning_engine: Optional LatentReasoningEngine instance
            context_engine: Optional ContextEngine instance
            config_path: Optional path to a configuration file
            config: Optional configuration dictionary (overrides file)
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        if config:
            for key, value in config.items():
                self.config_manager.set(key, value)
                
        # Store engines
        self.latent_reasoning_engine = latent_reasoning_engine
        self.context_engine = context_engine
        
        # Initialize components
        self.component_loader = ComponentLoader()
        self.error_handler = ErrorHandler(self.config_manager.get("logging"))
        self.handlers = {}
        
        # Initialize with error handling
        try:
            self._initialize_components()
            logger.info("ProcessorEngine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ProcessorEngine: {str(e)}")
            # Initialize with minimal components for graceful degradation
            self._initialize_minimal_components()
    
    def _initialize_components(self) -> None:
        """Initialize all required components with full functionality."""
        # Register model handlers
        self._register_default_handlers()
        
        # Initialize optimizer if available
        optimizer_module = import_optional(".optimizer", "ptolemy.multi_model_processor")
        if optimizer_module:
            self.optimizer = optimizer_module.Optimizer(self.config_manager.get("optimizer"))
        else:
            # Use simple optimizer as fallback
            self.optimizer = SimpleOptimizer(self.config_manager.get("optimizer"))
            
        # Initialize dispatcher if available
        dispatcher_module = import_optional(".dispatcher", "ptolemy.multi_model_processor")
        if dispatcher_module:
            self.dispatcher = dispatcher_module.Dispatcher(
                handlers=self.handlers,
                config=self.config_manager.get("dispatcher")
            )
        else:
            # Use simple dispatcher as fallback
            self.dispatcher = SimpleDispatcher(
                handlers=self.handlers,
                config=self.config_manager.get("dispatcher")
            )
            
        # Initialize metrics if available
        metrics_module = import_optional(".metrics", "ptolemy.multi_model_processor")
        if metrics_module:
            self.metrics = metrics_module.ProcessorMetrics(
                max_history=self.config_manager.get("metrics.max_history", 1000)
            )
        else:
            # Use simple metrics as fallback
            self.metrics = SimpleMetrics()
    
    def _initialize_minimal_components(self) -> None:
        """Initialize minimal components for graceful degradation."""
        # Register mock handler as fallback
        self._register_mock_handler()
        
        # Use simple components
        self.optimizer = SimpleOptimizer()
        self.dispatcher = SimpleDispatcher(handlers=self.handlers)
        self.metrics = SimpleMetrics()
        
        logger.info("ProcessorEngine initialized with minimal components")
    
    def _register_default_handlers(self) -> None:
        """Register all configured model handlers."""
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
        model_preference: Optional[str] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a task using the appropriate model.
        
        Args:
            task: The task description or query
            context: Optional context information
            model_preference: Optional preferred model to use
            options: Optional processing options
            
        Returns:
            Dictionary containing the processing results
        """
        # Create state for tracking
        state = ProcessorState.create(task=task, context=context)
        options = options or {}
        
        start_time = time.time()
        result = {
            "task_id": state.task_id,
            "success": False,
            "time": 0,
            "error": None
        }
        
        try:
            # Process the task with appropriate error handling
            state.advance_stage("initialization")
            
            # Retrieve additional context if context engine is available
            if self.context_engine and self.config_manager.get("processor.auto_context", True):
                try:
                    state.advance_stage("context_retrieval")
                    retrieved_context = await self.context_engine.retrieve_context(task)
                    if context is None:
                        context = {}
                    
                    # Merge retrieved context with provided context
                    if "retrieved" not in context:
                        context["retrieved"] = {}
                    context["retrieved"].update(retrieved_context)
                    state.context = context
                except Exception as e:
                    self.error_handler.handle_error(
                        e, 
                        context={"stage": "context_retrieval", "task_id": state.task_id},
                        severity=ErrorSeverity.WARNING
                    )
                    state.record_error(e, "context_retrieval", "warning")
                    logger.warning(f"Error retrieving context: {str(e)}")
            
            # Analyze task and determine parameters
            state.advance_stage("optimization")
            parameters = await self.optimizer.analyze_task(task, context)
            
            # Integrate with Latent Reasoning if available
            reasoning_result = None
            if self.latent_reasoning_engine and self.config_manager.get("latent_reasoning.enable", True):
                state.advance_stage("latent_reasoning")
                try:
                    # Use latent reasoning to enhance the task understanding
                    reasoning_iterations = options.get(
                        "reasoning_iterations", 
                        self.config_manager.get("latent_reasoning.default_iterations", 3)
                    )
                    
                    reasoning_result = await self.latent_reasoning_engine.process(
                        task=task,
                        context=context,
                        iterations=reasoning_iterations,
                        adaptive=self.config_manager.get("latent_reasoning.adaptive", True)
                    )
                    
                    state.reasoning_output = reasoning_result.get("output")
                    state.reasoning_iterations = reasoning_result.get("iterations", 0)
                    
                    # Update parameters based on reasoning output
                    if reasoning_result.get("output"):
                        parameters["reasoning_output"] = reasoning_result["output"]
                        
                except Exception as e:
                    self.error_handler.handle_error(
                        e, 
                        context={"stage": "latent_reasoning", "task_id": state.task_id},
                        severity=ErrorSeverity.WARNING
                    )
                    state.record_error(e, "latent_reasoning", "warning")
                    logger.warning(f"Error in latent reasoning: {str(e)}")
            
            # Select model
            state.advance_stage("model_selection")
            if model_preference and model_preference in self.handlers:
                selected_model = model_preference
                reason = "User preference"
            else:
                # Use optimizer to select the best model
                selected_model = await self.optimizer.select_optimal_model(
                    task=task,
                    parameters=parameters,
                    available_models=list(self.handlers.keys())
                )
                reason = "Automatic selection"
                
            state.select_model(selected_model, reason)
            
            # Optimize prompt
            state.advance_stage("prompt_optimization")
            optimized_prompt = await self.optimizer.optimize_prompt(
                task=task,
                context=context,
                reasoning_output=state.reasoning_output
            )
            
            # Dispatch to appropriate handler
            state.advance_stage("processing")
            dispatch_result = await self.dispatcher.dispatch(
                task=task,
                model=selected_model,
                prompt=optimized_prompt,
                parameters=parameters
            )
            
            # Record result
            state.advance_stage("completion")
            state.response = dispatch_result
            result.update({
                "output": dispatch_result.get("text", ""),
                "model": dispatch_result.get("model", selected_model),
                "tokens": dispatch_result.get("tokens", {}),
                "success": True,
                "state": state.to_dict()
            })
            
            # Record metrics
            await self._record_metrics(state, dispatch_result)
            
        except Exception as e:
            self.error_handler.handle_error(
                e, 
                context={"task_id": state.task_id, "stage": state.current_stage},
                severity=ErrorSeverity.ERROR
            )
            state.record_error(e, state.current_stage, "error")
            
            result["error"] = str(e)
            result["state"] = state.to_dict()
            
            # Attempt minimal processing if possible
            try:
                minimal_result = await self._minimal_processing(task, context)
                result["output"] = minimal_result.get("text", f"Error: {str(e)}")
                result["model"] = minimal_result.get("model", "fallback")
                result["minimal_fallback"] = True
            except Exception as fallback_error:
                logger.error(f"Minimal processing failed: {str(fallback_error)}")
                result["output"] = f"Unable to process task: {str(e)}"
        finally:
            # Calculate final timing
            process_time = time.time() - start_time
            result["time"] = process_time
            
        return result
    
    async def _minimal_processing(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform minimal processing when regular processing fails.
        
        Args:
            task: The task to process
            context: Optional context
            
        Returns:
            Minimal processing result
        """
        # Find the first working handler
        for model_id, handler in self.handlers.items():
            try:
                return await handler.process(
                    prompt=f"Process this task (minimal mode): {task}",
                    parameters={"minimal_mode": True}
                )
            except Exception:
                continue
                
        # If all handlers fail, return a basic response
        return {
            "text": f"Minimal processing: {task[:50]}...",
            "model": "text-only-fallback",
            "tokens": {"prompt": len(task.split()), "completion": 5}
        }
    
    async def _record_metrics(self, state: ProcessorState, result: Dict[str, Any]) -> None:
        """
        Record metrics for a processing task.
        
        Args:
            state: The processor state
            result: The processing result
        """
        # If metrics module is properly loaded, record detailed metrics
        if hasattr(self.metrics, "record_process"):
            try:
                tokens = result.get("tokens", {})
                input_tokens = tokens.get("prompt", 0)
                output_tokens = tokens.get("completion", 0)
                
                await self.metrics.record_process(
                    process_id=state.task_id,
                    task_type=state.context.get("task_type", "unknown") if state.context else "unknown",
                    model=result.get("model", "unknown"),
                    tokens_in=input_tokens,
                    tokens_out=output_tokens,
                    process_time=state.get_total_time(),
                    reasoning_iterations=state.reasoning_iterations,
                    success=True
                )
            except Exception as e:
                logger.error(f"Error recording metrics: {str(e)}")
        else:
            # Simple metrics recording
            self.metrics.record_task(state.task_id, state.get_total_time())
    
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


class SimpleOptimizer:
    """Simple optimizer for use when the full optimizer is not available."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the simple optimizer."""
        self.config = config or {}
        
    async def analyze_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze task complexity and determine optimal processing parameters.
        
        Args:
            task: Task description
            context: Optional context information
            
        Returns:
            Dictionary of optimal parameters
        """
        # Simple estimation based on task length
        words = len(task.split())
        
        return {
            "complexity": "medium" if words > 50 else "low",
            "estimated_tokens": words * 2,  # Simple estimation
            "use_reasoning": words > 100,  # Use reasoning for complex tasks
            "reasoning_iterations": 3
        }
        
    async def optimize_prompt(
        self, 
        task: str, 
        context: Optional[Dict[str, Any]] = None, 
        reasoning_output: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Optimize the prompt based on task and context.
        
        Args:
            task: Task description
            context: Context information
            reasoning_output: Optional output from latent reasoning
            
        Returns:
            Optimized prompt
        """
        prompt_parts = [f"Task: {task}"]
        
        # Add context if available
        if context:
            context_str = "\nContext:"
            
            if "retrieved" in context:
                retrieved = context["retrieved"]
                
                if "relationships" in retrieved and retrieved["relationships"]:
                    context_str += "\nRelationships:"
                    for rel in retrieved["relationships"][:3]:  # Limit to top 3
                        context_str += f"\n- {rel.get('name', 'Unnamed')}"
                        
                if "patterns" in retrieved and retrieved["patterns"]:
                    context_str += "\nPatterns:"
                    for pattern in retrieved["patterns"][:3]:  # Limit to top 3
                        context_str += f"\n- {pattern.get('name', 'Unnamed')}"
            
            prompt_parts.append(context_str)
        
        # Add reasoning output if available
        if reasoning_output:
            reasoning_str = "\nReasoning:"
            
            if isinstance(reasoning_output, dict):
                for key, value in reasoning_output.items():
                    if isinstance(value, str):
                        reasoning_str += f"\n{key}: {value}"
            elif isinstance(reasoning_output, str):
                reasoning_str += f"\n{reasoning_output}"
                
            prompt_parts.append(reasoning_str)
        
        # Add final instruction
        prompt_parts.append("\nPlease process this task thoroughly and provide a clear response.")
        
        return "\n".join(prompt_parts)
        
    async def select_optimal_model(
        self, 
        task: str, 
        parameters: Dict[str, Any], 
        available_models: Optional[List[str]] = None
    ) -> str:
        """
        Select the optimal model based on task requirements.
        
        Args:
            task: Task description
            parameters: Task parameters
            available_models: Available models to choose from
            
        Returns:
            Selected model identifier
        """
        if not available_models:
            return "mock-model"  # Default fallback
            
        # Simple selection - just take the first available model
        # In a real implementation, this would consider task type, complexity, etc.
        return available_models[0]


class SimpleDispatcher:
    """Simple dispatcher for use when the full dispatcher is not available."""
    
    def __init__(self, handlers: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simple dispatcher.
        
        Args:
            handlers: Dictionary of model handlers
            config: Optional configuration
        """
        self.handlers = handlers
        self.config = config or {}
        self.default_model = self.config.get("default_model", "mock-model")
        
    async def dispatch(
        self, 
        task: str, 
        model: str, 
        prompt: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dispatch a task to the appropriate handler.
        
        Args:
            task: Original task
            model: Selected model identifier
            prompt: Optimized prompt
            parameters: Processing parameters
            
        Returns:
            Processing result
        """
        parameters = parameters or {}
        
        # Get the handler for the selected model
        handler = self.handlers.get(model)
        
        # If handler not found, use default model
        if not handler and self.default_model in self.handlers:
            handler = self.handlers[self.default_model]
            model = self.default_model
            
        # If still no handler, raise error
        if not handler:
            raise ValueError(f"No handler available for model {model} and no default handler")
            
        # Process with the handler
        try:
            result = await handler.process(prompt, parameters)
            return result
        except Exception as e:
            logger.error(f"Error in model handler {model}: {str(e)}")
            
            # Try fallback if available
            if model != self.default_model and self.default_model in self.handlers:
                logger.info(f"Trying fallback model {self.default_model}")
                return await self.handlers[self.default_model].process(prompt, parameters)
            
            # Re-raise if no fallback available
            raise


class SimpleMetrics:
    """Simple metrics collector for use when the full metrics module is not available."""
    
    def __init__(self):
        """Initialize the simple metrics collector."""
        self.tasks = {}
        self.total_tasks = 0
        self.total_time = 0.0
        
    def record_task(self, task_id: str, process_time: float) -> None:
        """
        Record a task processing.
        
        Args:
            task_id: Task identifier
            process_time: Processing time in seconds
        """
        self.tasks[task_id] = {
            "time": process_time,
            "timestamp": time.time()
        }
        
        self.total_tasks += 1
        self.total_time += process_time
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary of simple metrics
        """
        return {
            "total_tasks": self.total_tasks,
            "total_time": self.total_time,
            "avg_time": self.total_time / max(1, self.total_tasks),
            "recent_tasks": len([t for t in self.tasks.values() if time.time() - t.get("timestamp", 0) < 3600])
        }