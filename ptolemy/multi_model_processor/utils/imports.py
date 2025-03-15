import importlib
import sys
from typing import Any, Dict, Callable, Optional, Type, TypeVar, cast
from loguru import logger

T = TypeVar('T')

def import_optional(module_name: str, package: Optional[str] = None, fallback: Any = None) -> Any:
    """
    Safely import a module, returning a fallback if it's not available.
    
    Args:
        module_name: Name of the module to import
        package: Package name for relative imports
        fallback: Optional fallback value or callable to return if import fails
        
    Returns:
        Imported module or fallback value
    """
    try:
        if module_name.startswith('.'):
            # Relative import
            if not package:
                # Determine package from caller
                frame = sys._getframe(1)
                caller_globals = frame.f_globals
                package = caller_globals.get('__name__', '').rsplit('.', 1)[0]
            
            module = importlib.import_module(module_name, package=package)
        else:
            # Absolute import
            module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        logger.warning(f"Optional module {module_name} not available: {str(e)}")
        if callable(fallback) and not isinstance(fallback, type):
            return fallback()
        return fallback


class ComponentLoader:
    """
    Manages loading of components with fallbacks for the Multi-Model Processor.
    
    This class provides a centralized way to load components with graceful
    fallbacks when dependencies are missing or errors occur.
    """
    
    def __init__(self):
        """Initialize the component loader."""
        self.loaded_components: Dict[str, Any] = {}
        self.fallbacks: Dict[str, Callable[..., Any]] = {}
        
    def register_fallback(self, component_name: str, fallback_factory: Callable[..., Any]) -> None:
        """
        Register a fallback for a component.
        
        Args:
            component_name: Name of the component
            fallback_factory: Factory function to create a fallback component
        """
        self.fallbacks[component_name] = fallback_factory
        
    def load_component(
        self, 
        component_name: str, 
        module_path: str, 
        class_name: str, 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        """
        Load a component with fallback support.
        
        Args:
            component_name: Name for the component
            module_path: Import path for the module
            class_name: Name of the class to instantiate
            *args: Positional arguments for the component constructor
            **kwargs: Keyword arguments for the component constructor
            
        Returns:
            Component instance or fallback
        """
        if component_name in self.loaded_components:
            return self.loaded_components[component_name]
            
        module = import_optional(module_path)
        if module is None:
            logger.warning(f"Could not load module {module_path} for component {component_name}")
            return self._use_fallback(component_name, *args, **kwargs)
            
        try:
            component_class = getattr(module, class_name)
            component = component_class(*args, **kwargs)
            self.loaded_components[component_name] = component
            logger.info(f"Successfully loaded component {component_name}")
            return component
        except (AttributeError, Exception) as e:
            logger.error(f"Error loading component {component_name}: {str(e)}")
            return self._use_fallback(component_name, *args, **kwargs)
            
    def load_component_class(
        self, 
        module_path: str, 
        class_name: str, 
        fallback_class: Optional[Type[T]] = None
    ) -> Type[T]:
        """
        Load a component class without instantiating it.
        
        Args:
            module_path: Import path for the module
            class_name: Name of the class to load
            fallback_class: Optional fallback class if the requested one isn't available
            
        Returns:
            Component class or fallback class
        """
        module = import_optional(module_path)
        if module is None:
            logger.warning(f"Could not load module {module_path} for class {class_name}")
            if fallback_class is not None:
                return fallback_class
            raise ImportError(f"Module {module_path} not found and no fallback provided")
            
        try:
            component_class = getattr(module, class_name)
            return cast(Type[T], component_class)
        except AttributeError as e:
            logger.error(f"Error loading class {class_name} from {module_path}: {str(e)}")
            if fallback_class is not None:
                return fallback_class
            raise AttributeError(f"Class {class_name} not found in {module_path} and no fallback provided")
            
    def _use_fallback(self, component_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Create and use a fallback component.
        
        Args:
            component_name: Name of the component
            *args: Positional arguments for the fallback constructor
            **kwargs: Keyword arguments for the fallback constructor
            
        Returns:
            Fallback component or None
        """
        if component_name not in self.fallbacks:
            logger.error(f"No fallback registered for component {component_name}")
            return None
            
        try:
            fallback = self.fallbacks[component_name](*args, **kwargs)
            logger.info(f"Using fallback for component {component_name}")
            self.loaded_components[component_name] = fallback
            return fallback
        except Exception as e:
            logger.error(f"Error creating fallback for {component_name}: {str(e)}")
            return None
