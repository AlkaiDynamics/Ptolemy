import pytest
import asyncio
from typing import Dict, Any
import uuid

from ptolemy.multi_model_processor.state import ProcessorState
from ptolemy.multi_model_processor.utils.error_handling import ErrorHandler, ErrorSeverity
from ptolemy.multi_model_processor.utils.config import ConfigManager
from ptolemy.multi_model_processor.utils.imports import import_optional, ComponentLoader
from ptolemy.multi_model_processor.handlers.mock_handler import MockModelHandler
from ptolemy.multi_model_processor.processor import ProcessorEngine


class TestProcessorState:
    """Tests for the ProcessorState class."""
    
    def test_create(self):
        """Test creating a new processor state."""
        state = ProcessorState.create(task="Test task")
        
        assert state.task == "Test task"
        assert state.task_id is not None
        assert state.created_at is not None
        assert state.current_stage == "created"
        assert len(state.stage_history) == 1
        
    def test_advance_stage(self):
        """Test advancing through stages."""
        state = ProcessorState.create(task="Test task")
        
        state.advance_stage("processing")
        assert state.current_stage == "processing"
        assert len(state.stage_history) == 2
        
        state.advance_stage("completion")
        assert state.current_stage == "completion"
        assert len(state.stage_history) == 3
        
    def test_select_model(self):
        """Test model selection tracking."""
        state = ProcessorState.create(task="Test task")
        
        state.select_model("gpt-4", "Best match for task")
        assert state.selected_model == "gpt-4"
        assert state.model_selection_reason == "Best match for task"
        
    def test_record_token_usage(self):
        """Test token usage recording."""
        state = ProcessorState.create(task="Test task")
        
        state.record_token_usage(input_tokens=100, output_tokens=50)
        assert state.token_usage["input"] == 100
        assert state.token_usage["output"] == 50
        assert state.token_usage["total"] == 150
        
    def test_record_error(self):
        """Test error recording."""
        state = ProcessorState.create(task="Test task")
        
        error = ValueError("Test error")
        state.record_error(error, "processing", "error")
        
        assert len(state.errors) == 1
        assert state.errors[0]["message"] == "Test error"
        assert state.errors[0]["stage"] == "processing"
        assert state.errors[0]["severity"] == "error"
        
    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = ProcessorState.create(task="Test task", context={"key": "value"})
        state.advance_stage("processing")
        state.select_model("gpt-4", "Best match")
        state.record_token_usage(100, 50)
        
        state_dict = state.to_dict()
        
        assert state_dict["task"] == "Test task"
        assert state_dict["context"] == {"key": "value"}
        assert state_dict["current_stage"] == "processing"
        assert state_dict["selected_model"] == "gpt-4"
        assert state_dict["token_usage"]["total"] == 150


class TestErrorHandler:
    """Tests for the ErrorHandler class."""
    
    def test_handle_error_info(self, error_handler):
        """Test handling info level errors."""
        error = ValueError("Test info error")
        error_handler.handle_error(error, {"test": "context"}, ErrorSeverity.INFO)
        # Info errors are suppressed by default in test config
        # This test just verifies no exception is raised
        
    def test_handle_error_warning(self, error_handler):
        """Test handling warning level errors."""
        error = ValueError("Test warning error")
        error_handler.handle_error(error, {"test": "context"}, ErrorSeverity.WARNING)
        # Just verifies no exception is raised
        
    def test_handle_error_propagation(self, error_handler):
        """Test error propagation for errors."""
        error = ValueError("Test error to propagate")
        
        # Test with propagate=True (default for ERROR severity)
        with pytest.raises(ValueError):
            error_handler.handle_error(error, propagate=True)


class TestConfigManager:
    """Tests for the ConfigManager class."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = ConfigManager()
        
        # Check some default values
        assert config.get("processor.enable_caching") is True
        assert config.get("logging.level") == "INFO"
        
    def test_get_with_override(self, config_manager, monkeypatch):
        """Test getting config with environment override."""
        # Set environment variable override
        monkeypatch.setenv("PTOLEMY_PROCESSOR_DEFAULT_MODEL", "env-model")
        
        # Should use the environment variable value
        assert config_manager.get("processor.default_model") == "env-model"
        
    def test_set_and_get(self):
        """Test setting and getting configuration values."""
        config = ConfigManager()
        
        config.set("test.nested.value", 42)
        assert config.get("test.nested.value") == 42
        
    def test_get_all(self, config_manager):
        """Test getting all configuration."""
        config_dict = config_manager.get_all()
        
        assert "processor" in config_dict
        assert "models" in config_dict
        assert config_dict["processor"]["max_tokens"] == 1000


class TestMockModelHandler:
    """Tests for the MockModelHandler class."""
    
    @pytest.mark.asyncio
    async def test_process(self):
        """Test processing a prompt."""
        handler = MockModelHandler({"latency": 0.1})  # Low latency for fast tests
        
        result = await handler.process("Test prompt")
        
        assert "text" in result
        assert "model" in result
        assert "tokens" in result
        assert result["tokens"]["prompt"] > 0
        assert result["tokens"]["completion"] > 0
        
    @pytest.mark.asyncio
    async def test_get_capabilities(self):
        """Test getting model capabilities."""
        handler = MockModelHandler()
        
        capabilities = await handler.get_capabilities()
        
        assert "max_tokens" in capabilities
        assert "strengths" in capabilities
        assert "weaknesses" in capabilities
        assert len(capabilities["strengths"]) > 0
        
    @pytest.mark.asyncio
    async def test_predefined_response(self):
        """Test using predefined responses."""
        handler = MockModelHandler({
            "predefined_responses": {
                "special_keyword": "This is a predefined response"
            },
            "latency": 0.1
        })
        
        result = await handler.process("This has a special_keyword in it")
        
        assert result["text"] == "This is a predefined response"
        assert result["metadata"]["predefined"] is True


class TestProcessorIntegration:
    """Integration tests for the processor components."""
    
    @pytest.mark.asyncio
    async def test_processor_basic_flow(self, mock_latent_reasoning_engine, mock_context_engine):
        """Test basic processing flow with mock components."""
        # Create processor with mock components
        processor = ProcessorEngine(
            latent_reasoning_engine=mock_latent_reasoning_engine,
            context_engine=mock_context_engine,
            config={
                "processor": {
                    "default_model": "mock-model",
                    "auto_context": True
                },
                "models": {
                    "mock-model": {
                        "handler_type": "mock",
                        "latency": 0.1  # Fast for testing
                    }
                },
                "latent_reasoning": {
                    "enable": True,
                    "default_iterations": 2
                }
            }
        )
        
        # Process a task
        result = await processor.process_task(
            task="Test integration task",
            context={"user_id": "test-user"}
        )
        
        # Verify result structure
        assert result["success"] is True
        assert "output" in result
        assert "model" in result
        assert "tokens" in result
        assert "state" in result
        
        # Verify processor state in result
        state = result["state"]
        assert state["task"] == "Test integration task"
        assert state["current_stage"] == "completion"
        assert len(state["stage_history"]) > 3  # Should have gone through multiple stages
        assert "latent_reasoning" in [s["stage"] for s in state["stage_history"]]
        assert state["context"]["user_id"] == "test-user"
        
    @pytest.mark.asyncio
    async def test_get_available_models(self):
        """Test getting available models list."""
        processor = ProcessorEngine(
            config={
                "models": {
                    "mock-model-1": {"handler_type": "mock"},
                    "mock-model-2": {"handler_type": "mock"}
                }
            }
        )
        
        models = await processor.get_available_models()
        
        assert len(models) > 0
        assert "id" in models[0]
        assert "capabilities" in models[0]
        
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, monkeypatch):
        """Test error handling with fallback processing."""
        # Create a processor with a mock handler that will fail
        processor = ProcessorEngine(
            config={
                "models": {
                    "error-model": {
                        "handler_type": "mock",
                        "error_rate": 1.0  # Will always fail
                    },
                    "fallback-model": {
                        "handler_type": "mock",
                        "error_rate": 0.0  # Will never fail
                    }
                },
                "processor": {
                    "default_model": "fallback-model"
                }
            }
        )
        
        # Monkeypatch the optimizer to always select the error model first
        async def mock_select(*args, **kwargs):
            return "error-model"
            
        monkeypatch.setattr(processor.optimizer, "select_optimal_model", mock_select)
        
        # Process a task - should fail but use fallback
        result = await processor.process_task("Test error handling")
        
        # Should still get a result due to fallback
        assert "output" in result
        assert "error" in result  # But should have an error recorded
        assert "minimal_fallback" in result  # And should indicate fallback was used
        assert result["model"] != "error-model"  # Should not use the error model
