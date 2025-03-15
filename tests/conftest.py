import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Common mock for OpenAI client
@pytest.fixture
def mock_openai_client():
    """Mock for OpenAI client with common methods."""
    mock_client = AsyncMock()
    
    # Mock chat completions
    mock_completion = AsyncMock()
    mock_completion.choices = [AsyncMock()]
    mock_completion.choices[0].message.content = "Mocked response"
    
    # Set up the mock client structure
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    return mock_client

# Helper to mock all API providers
@pytest.fixture
def mock_all_providers(monkeypatch):
    """Mock all AI provider clients."""
    # Create mocks for each provider
    providers = [
        "openai", "anthropic", "mistral", "groq", "cohere", 
        "ollama", "lmstudio", "openrouter", "deepseek", "google",
        "xai", "qwen", "microsoft"
    ]
    
    mocks = {}
    for provider in providers:
        mock = AsyncMock()
        mocks[provider] = mock
        monkeypatch.setattr(f"ptolemy.multi_model.MultiModelProcessor.ai_clients.{provider}", mock)
    
    return mocks

# Fixture for test database
@pytest.fixture(scope="session")
def test_db_url():
    """Return a test database URL."""
    return "sqlite+aiosqlite:///data/test_ptolemy.db"

# Fixture to patch settings
@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    mock_settings = MagicMock()
    mock_settings.database_url = "sqlite+aiosqlite:///data/test_ptolemy.db"
    mock_settings.max_retries = 3
    mock_settings.base_delay = 0.01
    mock_settings.default_provider = "test_provider"
    mock_settings.log_level = "DEBUG"
    
    # Patch the settings import
    with patch("ptolemy.database.settings", mock_settings):
        with patch("ptolemy.temporal_core.settings", mock_settings):
            with patch("ptolemy.context_engine.settings", mock_settings):
                with patch("ptolemy.multi_model.settings", mock_settings):
                    yield mock_settings

# Fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
