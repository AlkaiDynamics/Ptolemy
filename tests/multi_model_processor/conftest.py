import pytest
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import json
import uuid

from ptolemy.multi_model_processor.utils.config import ConfigManager
from ptolemy.multi_model_processor.utils.error_handling import ErrorHandler
from ptolemy.multi_model_processor.state import ProcessorState

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Fixture providing a test configuration."""
    return {
        "processor": {
            "default_model": "mock-model",
            "max_tokens": 1000,
            "enable_caching": False
        },
        "models": {
            "mock-model": {
                "handler_type": "mock",
                "capabilities": {
                    "max_tokens": 2000,
                    "supports_streaming": True,
                    "strengths": ["testing", "development"]
                }
            }
        },
        "logging": {
            "level": "DEBUG"
        }
    }

@pytest.fixture
def config_manager(test_config) -> ConfigManager:
    """Fixture providing a config manager with test configuration."""
    manager = ConfigManager()
    
    # Manually set the configuration values
    for key, value in test_config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                manager.set(f"{key}.{sub_key}", sub_value)
        else:
            manager.set(key, value)
            
    return manager

@pytest.fixture
def error_handler() -> ErrorHandler:
    """Fixture providing an error handler for testing."""
    config = {
        "suppress_levels": ["DEBUG", "INFO"],
        "log_level": "DEBUG",
        "include_traceback": True
    }
    return ErrorHandler(config)

@pytest.fixture
def processor_state() -> ProcessorState:
    """Fixture providing a processor state for testing."""
    return ProcessorState.create(
        task="This is a test task",
        context={"test_key": "test_value"}
    )

@pytest.fixture
def mock_latent_reasoning_engine():
    """Fixture providing a mock latent reasoning engine."""
    class MockLatentReasoningEngine:
        async def process(self, task, context=None, iterations=None, adaptive=False):
            """Mock implementation of the process method."""
            return {
                "output": f"Processed: {task}",
                "iterations": iterations or 3,
                "adaptive_stopped": adaptive,
                "convergence": 0.001 if adaptive else None
            }
            
        async def calculate_change(self, previous_state, current_state):
            """Mock implementation of calculate_change method."""
            return 0.001
            
    return MockLatentReasoningEngine()

@pytest.fixture
def mock_context_engine():
    """Fixture providing a mock context engine."""
    class MockContextEngine:
        async def retrieve_context(self, query, limit=5):
            """Mock implementation of retrieve_context method."""
            return {
                "relationships": [
                    {"id": str(uuid.uuid4()), "name": f"Test Relationship {i}", "data": {"value": i}} 
                    for i in range(2)
                ],
                "patterns": [
                    {"id": str(uuid.uuid4()), "name": f"Test Pattern {i}", "data": {"value": i}}
                    for i in range(2)
                ],
                "insights": [
                    {"id": str(uuid.uuid4()), "name": f"Test Insight {i}", "data": {"value": i}}
                    for i in range(1)
                ]
            }
            
        async def retrieve_relationships(self, query, limit=5):
            """Mock implementation of retrieve_relationships method."""
            return [
                {"id": str(uuid.uuid4()), "name": f"Test Relationship {i}", "data": {"value": i}} 
                for i in range(limit)
            ]
            
    return MockContextEngine()

@pytest.fixture
def sample_task() -> str:
    """Fixture providing a sample task for testing."""
    return "Analyze the performance implications of increasing the cache size in the database layer."

@pytest.fixture
def sample_context() -> Dict[str, Any]:
    """Fixture providing a sample context for testing."""
    return {
        "user_preferences": {
            "detail_level": "high",
            "format": "markdown"
        },
        "project_info": {
            "name": "PTOLEMY",
            "description": "Knowledge management and reasoning system"
        },
        "relevant_data": [
            {"type": "code_snippet", "content": "def process_data(data):\n    return data.transform()"},
            {"type": "documentation", "content": "The database layer supports various cache configurations."}
        ]
    }
