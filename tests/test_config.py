import pytest
import os
from unittest.mock import patch
import tempfile
import shutil

from ptolemy.config import settings

@pytest.fixture
def temp_settings_dir():
    """Create a temporary directory for settings files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create a temporary settings file
    with open(os.path.join(temp_dir, "settings.toml"), "w") as f:
        f.write("""
        [default]
        LOG_LEVEL = "INFO"
        DEFAULT_PROVIDER = "test_provider"
        
        [dev]
        LOG_LEVEL = "DEBUG"
        
        [prod]
        LOG_LEVEL = "WARNING"
        """)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_settings_default_values():
    """Test that default settings are loaded correctly."""
    assert hasattr(settings, "LOG_LEVEL")
    assert settings.get("DEFAULT_PROVIDER") is not None

@patch.dict(os.environ, {"PTOLEMY_ENV": "dev"})
def test_settings_environment_override():
    """Test that environment-specific settings override defaults."""
    # Force reload settings
    from dynaconf import Dynaconf
    test_settings = Dynaconf(
        settings_files=["settings.toml"],
        environments=True,
        env_switcher="PTOLEMY_ENV",
        load_dotenv=True,
        envvar_prefix="PTOLEMY"
    )
    
    # Check that development environment is active
    assert test_settings.LOG_LEVEL == "DEBUG"

@patch.dict(os.environ, {"PTOLEMY_LOG_LEVEL": "ERROR"})
def test_settings_env_var_override():
    """Test that environment variables override settings file values."""
    # Force reload settings
    from dynaconf import Dynaconf
    test_settings = Dynaconf(
        settings_files=["settings.toml"],
        environments=True,
        env_switcher="PTOLEMY_ENV",
        load_dotenv=True,
        envvar_prefix="PTOLEMY"
    )
    
    # Check that the environment variable overrides the setting
    assert test_settings.LOG_LEVEL == "ERROR"

def test_settings_file_loading(temp_settings_dir):
    """Test loading settings from a specific file."""
    settings_file = os.path.join(temp_settings_dir, "settings.toml")
    
    # Create a test settings instance
    from dynaconf import Dynaconf
    test_settings = Dynaconf(
        settings_files=[settings_file],
        environments=True,
        env_switcher="PTOLEMY_ENV",
        load_dotenv=True,
        envvar_prefix="PTOLEMY"
    )
    
    # Check that settings from the file are loaded
    assert test_settings.DEFAULT_PROVIDER == "test_provider"
    assert test_settings.LOG_LEVEL == "INFO"

def test_settings_type_conversion():
    """Test that settings are converted to the appropriate types."""
    # Set an integer value
    with patch.dict(os.environ, {"PTOLEMY_MAX_RETRIES": "3"}):
        from dynaconf import Dynaconf
        test_settings = Dynaconf(
            settings_files=["settings.toml"],
            environments=True,
            env_switcher="PTOLEMY_ENV",
            load_dotenv=True,
            envvar_prefix="PTOLEMY"
        )
        
        # Check that the value is converted to an integer
        assert isinstance(test_settings.MAX_RETRIES, int)
        assert test_settings.MAX_RETRIES == 3
    
    # Set a boolean value
    with patch.dict(os.environ, {"PTOLEMY_DEBUG": "true"}):
        from dynaconf import Dynaconf
        test_settings = Dynaconf(
            settings_files=["settings.toml"],
            environments=True,
            env_switcher="PTOLEMY_ENV",
            load_dotenv=True,
            envvar_prefix="PTOLEMY"
        )
        
        # Check that the value is converted to a boolean
        assert isinstance(test_settings.DEBUG, bool)
        assert test_settings.DEBUG is True
