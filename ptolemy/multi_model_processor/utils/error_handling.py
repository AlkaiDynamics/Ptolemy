import traceback
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import logging
from loguru import logger

class ErrorSeverity(Enum):
    """
    Severity levels for errors encountered during multi-model processing.
    
    Levels:
    - DEBUG: Information useful for debugging
    - INFO: Normal but significant events
    - WARNING: Potential issues that don't prevent operation
    - ERROR: Errors that prevent a specific operation
    - CRITICAL: Critical errors that prevent system functionality
    """
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class ErrorHandler:
    """
    Handles errors with appropriate propagation based on severity.
    
    This class provides a centralized error handling system for the Multi-Model Processor,
    with configurable error suppression, callbacks, and logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the error handler.
        
        Args:
            config: Optional configuration dictionary with settings like:
                   - suppress_levels: List of ErrorSeverity levels to suppress
                   - log_format: Format for log messages
                   - include_traceback: Whether to include tracebacks in logs
        """
        self.config = config or {}
        self.suppress_levels = self._parse_suppress_levels(
            self.config.get("suppress_levels", [])
        )
        self.error_callbacks = []
        self.include_traceback = self.config.get("include_traceback", True)
        
        # Configure logging if not already configured
        if not self.config.get("skip_logging_config", False):
            self._configure_logging()
    
    def _parse_suppress_levels(self, levels) -> List[ErrorSeverity]:
        """
        Parse suppress levels from configuration.
        
        Args:
            levels: List of level names or ErrorSeverity enum values
            
        Returns:
            List of ErrorSeverity enum values
        """
        result = []
        for level in levels:
            if isinstance(level, ErrorSeverity):
                result.append(level)
            elif isinstance(level, str) and hasattr(ErrorSeverity, level.upper()):
                result.append(getattr(ErrorSeverity, level.upper()))
        return result
    
    def _configure_logging(self) -> None:
        """Configure logging with appropriate format and levels."""
        log_format = self.config.get(
            "log_format", 
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Configure loguru
        logger.configure(
            handlers=[
                {
                    "sink": logging.StreamHandler(),
                    "format": log_format,
                    "level": self.config.get("log_level", "INFO"),
                }
            ]
        )
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for error handling.
        
        Args:
            callback: Function that takes an error_info dictionary
        """
        self.error_callbacks.append(callback)
    
    def handle_error(
        self,
        error: Exception, 
        context: Optional[Dict[str, Any]] = None, 
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ) -> bool:
        """
        Handle an error with appropriate propagation.
        
        Args:
            error: The error that occurred
            context: Additional context about the error
            severity: Error severity level
            
        Returns:
            True if error should be propagated, False if suppressed
        """
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc() if self.include_traceback else None,
            "context": context or {},
            "severity": severity.name,
            "severity_level": severity.value
        }
        
        # Log the error
        self._log_error(error_info)
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.warning(f"Error in error callback: {str(e)}")
        
        # Determine if error should be propagated
        if severity in self.suppress_levels:
            return False
        
        # Return True to indicate error should be propagated
        return True
    
    def _log_error(self, error_info: Dict[str, Any]) -> None:
        """
        Log an error with the appropriate log level.
        
        Args:
            error_info: Information about the error
        """
        severity = error_info["severity"]
        message = f"{error_info['type']}: {error_info['message']}"
        
        # Add context to the message if available
        context_str = ""
        if error_info["context"]:
            context_items = [f"{k}={v}" for k, v in error_info["context"].items()]
            context_str = f" [{', '.join(context_items)}]"
        
        full_message = f"{message}{context_str}"
        
        # Log with appropriate level
        if severity == ErrorSeverity.DEBUG.name:
            logger.debug(full_message)
        elif severity == ErrorSeverity.INFO.name:
            logger.info(full_message)
        elif severity == ErrorSeverity.WARNING.name:
            logger.warning(full_message)
        elif severity == ErrorSeverity.ERROR.name:
            logger.error(full_message)
        elif severity == ErrorSeverity.CRITICAL.name:
            logger.critical(full_message)
            
        # Log traceback at debug level if available
        if error_info["traceback"] and severity in [
            ErrorSeverity.ERROR.name, 
            ErrorSeverity.CRITICAL.name
        ]:
            logger.debug(f"Traceback for {error_info['type']}:\n{error_info['traceback']}")
    
    def create_context_error_handler(self, base_context: Dict[str, Any]):
        """
        Create an error handler with a base context.
        
        Args:
            base_context: Base context to include with all errors
            
        Returns:
            Function that handles errors with the base context
        """
        def handle_with_context(error, additional_context=None, severity=ErrorSeverity.ERROR):
            context = base_context.copy()
            if additional_context:
                context.update(additional_context)
            return self.handle_error(error, context, severity)
        
        return handle_with_context
