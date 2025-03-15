import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger

class ConfigManager:
    """
    Manages configuration across the Multi-Model Processor.
    
    This class provides a centralized configuration system with:
    - Default configuration values
    - Configuration file loading
    - Environment variable overrides
    - Nested key access
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        self.config: Dict[str, Any] = self._load_defaults()
        
        # Load configuration from file if provided
        if config_path:
            file_config = self._load_config(config_path)
            self._deep_update(self.config, file_config)
            
        # Load environment overrides
        self.environment_overrides = self._load_environment_overrides()
        
    def _load_defaults(self) -> Dict[str, Any]:
        """
        Load default configuration values.
        
        Returns:
            Dictionary with default configuration
        """
        return {
            "processor": {
                "default_model": "auto",
                "max_tokens": 2048,
                "enable_caching": True,
                "cache_ttl": 3600,  # 1 hour
                "adaptive_routing": True
            },
            "models": {
                # Empty by default, will be populated from config file or env vars
            },
            "logging": {
                "level": "INFO",
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
                "include_traceback": True
            },
            "latent_reasoning": {
                "enable": True,
                "default_iterations": 3,
                "convergence_threshold": 0.01
            },
            "security": {
                "mask_credentials_in_logs": True,
                "use_keyring": False
            }
        }
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Configuration file not found: {config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            return {}
            
    def _load_environment_overrides(self) -> Dict[str, str]:
        """
        Load configuration overrides from environment variables.
        
        Returns:
            Dictionary of environment overrides
        """
        overrides = {}
        prefix = "PTOLEMY_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                overrides[key] = value
                
        return overrides
        
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a target dictionary with values from a source dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with environment override support.
        
        Args:
            key: Configuration key in dot notation (e.g., 'processor.default_model')
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        # Check environment override
        env_key = f"PTOLEMY_{key.upper().replace('.', '_')}"
        if env_key in self.environment_overrides:
            return self._convert_type(self.environment_overrides[env_key])
        
        # Check config dictionary
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def _convert_type(self, value: str) -> Any:
        """
        Convert string environment variable to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value
        """
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
            
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
            
        # Try to convert to boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
            
        # Try to convert to JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
            
        # Return as string
        return value
        
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to the nested dictionary
        for i, k in enumerate(keys[:-1]):
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
            
        # Set the value
        config_ref[keys[-1]] = value
        
    def get_all(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Complete configuration with environment overrides applied
        """
        # Create a copy of the config
        config_copy = json.loads(json.dumps(self.config))
        
        # Apply environment overrides
        for env_key, value in self.environment_overrides.items():
            if env_key.startswith("PTOLEMY_"):
                config_key = env_key[8:].lower().replace('_', '.')  # Remove prefix and convert format
                keys = config_key.split('.')
                
                # Navigate to the correct nested dictionary
                config_ref = config_copy
                for i, k in enumerate(keys[:-1]):
                    if k not in config_ref:
                        config_ref[k] = {}
                    config_ref = config_ref[k]
                    
                # Set the value
                config_ref[keys[-1]] = self._convert_type(value)
                
        return config_copy
        
    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
