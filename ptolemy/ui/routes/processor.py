"""
Routes for the Multi-Model Processor UI and API.

These routes handle the UI rendering and API endpoints for the
Multi-Model Processor, providing access to processor capabilities,
cache statistics, and error analysis.
"""
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Callable, TypeVar, Awaitable
import asyncio
import json
from loguru import logger
import time
import os
from pathlib import Path
import traceback
from functools import wraps
import uuid

from ptolemy.multi_model_processor.processor import ProcessorEngine
from ptolemy.multi_model_processor.utils.caching import ResponseCache
from ptolemy.multi_model_processor.utils.performance import PerformanceMonitor

# Define custom exceptions for better error handling
class ConfigurationError(Exception):
    """Error raised when there's a configuration issue."""
    pass

class ModelNotAvailableError(Exception):
    """Error raised when a requested model is not available."""
    pass

class TaskProcessingError(Exception):
    """Error raised during task processing."""
    pass

# Type for routes that return a response
T = TypeVar('T')

# API error handler decorator
def api_error_handler(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator for consistent API error handling.
    
    Args:
        func: Async function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())
        logger.info(f"Request {request_id}: Starting {func.__name__}")
        try:
            response = await func(*args, **kwargs)
            return response
        except ConfigurationError as e:
            logger.error(f"Request {request_id}: Configuration error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "configuration_error", 
                    "message": "Service misconfigured",
                    "details": str(e)
                }
            )
        except ModelNotAvailableError as e:
            logger.warning(f"Request {request_id}: Model not available: {str(e)}")
            available_models = []
            try:
                processor = get_processor()
                available_models = processor.get_model_handlers()
            except Exception:
                pass
                
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "model_unavailable", 
                    "message": "Requested model is unavailable",
                    "details": str(e),
                    "available_models": available_models
                }
            )
        except TaskProcessingError as e:
            logger.error(f"Request {request_id}: Task processing error: {str(e)}")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "processing_error", 
                    "message": "Error processing the task",
                    "details": str(e)
                }
            )
        except Exception as e:
            logger.critical(f"Request {request_id}: Unhandled exception: {str(e)}", exc_info=True)
            logger.critical(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "internal_error", 
                    "message": "An unexpected error occurred",
                    "details": str(e) if os.getenv("DEBUG", "False").lower() == "true" else None
                }
            )
    return wrapper

# Models for request/response
class ProcessRequest(BaseModel):
    """Request model for process task."""
    task: str
    model_preference: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    bypass_cache: Optional[bool] = False

class ProcessResponse(BaseModel):
    """Response model for process task."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None
    cached: Optional[bool] = False
    processing_time: Optional[float] = None
    state: Optional[Dict[str, Any]] = None

# Initialize router
router = APIRouter(prefix="/processor", tags=["processor"])

# Initialize templates
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Singleton processor instance
_processor_instance = None

def get_processor() -> ProcessorEngine:
    """
    Get or create the processor singleton instance.
    
    Returns:
        ProcessorEngine instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        try:
            # Load configuration
            config = {
                "processor": {
                    "enable_caching": True,
                    "default_model": "anthropic_claude"
                },
                "cache": {
                    "max_size": 1000,
                    "default_ttl": 3600,
                    "eviction_policy": "lru",
                    "performance_monitoring": True
                },
                "error_recovery": {
                    "circuit_breaker_threshold": 3,
                    "circuit_breaker_timeout": 60.0,
                    "max_retry_count": 3,
                    "fallback_models": ["openai_gpt4", "anthropic_claude"]
                },
                "models": {
                    "anthropic_claude": {
                        "handler_type": "anthropic",
                        "config": {
                            "model": "claude-3-opus-20240229",
                            "api_key": os.environ.get("ANTHROPIC_API_KEY")
                        }
                    },
                    "openai_gpt4": {
                        "handler_type": "openai",
                        "config": {
                            "model": "gpt-4-turbo",
                            "api_key": os.environ.get("OPENAI_API_KEY")
                        }
                    }
                }
            }
            
            # Create processor instance
            _processor_instance = ProcessorEngine(config=config)
            logger.info("Created processor singleton instance")
        except Exception as e:
            logger.error(f"Error creating processor instance: {str(e)}")
            logger.error(traceback.format_exc())
            # Create a minimal processor for testing
            _processor_instance = ProcessorEngine()
    
    return _processor_instance

# UI Route
@router.get("", response_class=HTMLResponse)
async def get_processor_ui(request: Request):
    """
    Render the processor UI.
    
    Args:
        request: HTTP request
        
    Returns:
        HTML response with processor UI
    """
    return templates.TemplateResponse(
        "processor.html",
        {"request": request}
    )

# API Routes
@router.get("/models", response_class=JSONResponse)
@api_error_handler
async def get_models():
    """
    Get available models.
    
    Returns:
        JSON response with models list
    """
    processor = get_processor()
    models = processor.get_model_handlers()
    return {"models": models}

@router.post("/process", response_model=ProcessResponse)
@api_error_handler
async def process_task(request: ProcessRequest):
    """
    Process a task.
    
    Args:
        request: Process request
        
    Returns:
        Process response
    """
    # Validate input
    if not request.task or not isinstance(request.task, str):
        raise TaskProcessingError("Task must be a non-empty string")
    
    start_time = time.time()
    processor = get_processor()
    
    # Process the task
    result = await processor.process_task(
        task=request.task,
        model_preference=request.model_preference,
        parameters=request.parameters,
        bypass_cache=request.bypass_cache
    )
    
    # Check for errors
    if not result.get("success", False):
        raise TaskProcessingError(result.get("error", "Unknown error"))
    
    # Return successful response
    return ProcessResponse(
        success=True,
        output=result.get("output", ""),
        model=result.get("model"),
        cached=result.get("cached", False),
        processing_time=result.get("processing_time", time.time() - start_time),
        state=result.get("state")
    )

@router.post("/process-stream")
@api_error_handler
async def process_task_stream(request: ProcessRequest):
    """
    Process a task with streaming response.
    
    Args:
        request: Process request
        
    Returns:
        Streaming response
    """
    # Validate input
    if not request.task or not isinstance(request.task, str):
        raise TaskProcessingError("Task must be a non-empty string")
        
    async def stream_generator():
        processor = get_processor()
        
        # Process task with streaming
        async for chunk in processor.process_task_stream(
            task=request.task,
            model_preference=request.model_preference,
            parameters=request.parameters
        ):
            # Format as SSE
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

@router.get("/cache-stats", response_class=JSONResponse)
@api_error_handler
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        JSON response with cache statistics
    """
    processor = get_processor()
    stats = processor.get_cache_stats()
    return stats

@router.post("/clear-cache", response_class=JSONResponse)
@api_error_handler
async def clear_cache():
    """
    Clear the processor cache.
    
    Returns:
        JSON response with success status
    """
    processor = get_processor()
    if processor.cache:
        processor.cache.clear()
        return {"success": True, "message": "Cache cleared successfully"}
    else:
        return {"success": False, "message": "Cache is not enabled"}

@router.get("/error-analysis", response_class=JSONResponse)
@api_error_handler
async def get_error_analysis():
    """
    Get error analysis.
    
    Returns:
        JSON response with error analysis
    """
    processor = get_processor()
    analysis = processor.get_error_analysis()
    return analysis

@router.get("/api-specs", response_class=JSONResponse)
@api_error_handler
async def get_api_specs():
    """
    Return OpenAPI-compatible specs for frontend integration.
    
    Returns:
        JSON response with API specifications
    """
    processor = get_processor()
    available_models = processor.get_model_handlers()
    
    return {
        "endpoints": {
            "process": {
                "url": "/processor/process",
                "method": "POST",
                "parameters": {
                    "task": "string",
                    "model_preference": "string?",
                    "parameters": "object?",
                    "bypass_cache": "boolean?"
                }
            },
            "process_stream": {
                "url": "/processor/process-stream",
                "method": "POST",
                "parameters": {
                    "task": "string",
                    "model_preference": "string?",
                    "parameters": "object?"
                }
            },
            "models": {
                "url": "/processor/models",
                "method": "GET"
            },
            "cache_stats": {
                "url": "/processor/cache-stats",
                "method": "GET"
            },
            "clear_cache": {
                "url": "/processor/clear-cache",
                "method": "POST"
            },
            "error_analysis": {
                "url": "/processor/error-analysis",
                "method": "GET"
            }
        },
        "models": available_models,
        "version": "1.0.0"
    }
