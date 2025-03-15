import os
from pathlib import Path
import sys
from loguru import logger
from dynaconf import Dynaconf

# Configure logger
logger.remove()
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO").upper())
logger.add("logs/ptolemy.log", rotation="10 MB", retention="1 week", level="DEBUG")

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPORAL_DIR = DATA_DIR / "temporal"
CONTEXT_DIR = DATA_DIR / "context"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
(CONTEXT_DIR / "relationships").mkdir(exist_ok=True)
(CONTEXT_DIR / "patterns").mkdir(exist_ok=True)
(CONTEXT_DIR / "insights").mkdir(exist_ok=True)

# Initialize Dynaconf
settings = Dynaconf(
    settings_files=[
        f"{BASE_DIR}/settings.toml",  # Main settings file
        f"{BASE_DIR}/.secrets.toml",  # Secret settings (API keys, etc.)
    ],
    environments=True,  # Enable environment-specific settings
    env_switcher="PTOLEMY_ENV",  # Environment variable to switch environments
    load_dotenv=True,  # Load .env file
)

# AI Provider Configurations
AI_PROVIDERS = {
    "openai": {
        "api_key": settings.get("OPENAI_API_KEY", ""),
        "available_models": settings.get("OPENAI_MODELS", [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        ]),
        "default_model": settings.get("DEFAULT_MODEL", "gpt-4"),
    },
    "anthropic": {
        "api_key": settings.get("ANTHROPIC_API_KEY", ""),
        "available_models": settings.get("ANTHROPIC_MODELS", [
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
        ]),
        "default_model": settings.get("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229"),
    },
    "mistral": {
        "api_key": settings.get("MISTRAL_API_KEY", ""),
        "available_models": settings.get("MISTRAL_MODELS", [
            "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large-latest"
        ]),
        "default_model": settings.get("MISTRAL_DEFAULT_MODEL", "mistral-medium"),
    },
    "groq": {
        "api_key": settings.get("GROQ_API_KEY", ""),
        "available_models": settings.get("GROQ_MODELS", [
            "llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"
        ]),
        "default_model": settings.get("GROQ_DEFAULT_MODEL", "mixtral-8x7b-32768"),
    },
    "cohere": {
        "api_key": settings.get("COHERE_API_KEY", ""),
        "available_models": settings.get("COHERE_MODELS", [
            "command", "command-light", "command-nightly", "command-r", "command-r-plus"
        ]),
        "default_model": settings.get("COHERE_DEFAULT_MODEL", "command-r"),
    },
    "ollama": {
        "api_base": settings.get("OLLAMA_API_BASE", "http://localhost:11434"),
        "available_models": settings.get("OLLAMA_MODELS", [
            "llama2", "llama2:13b", "llama2:70b", "mistral", "mixtral", "phi", "gemma", "codellama"
        ]),
        "default_model": settings.get("OLLAMA_DEFAULT_MODEL", "mixtral"),
    },
    "lmstudio": {
        "api_base": settings.get("LMSTUDIO_API_BASE", "http://localhost:1234/v1"),
        "available_models": settings.get("LMSTUDIO_MODELS", ["local-model"]),
        "default_model": settings.get("LMSTUDIO_DEFAULT_MODEL", "local-model"),
    },
    "openrouter": {
        "api_key": settings.get("OPENROUTER_API_KEY", ""),
        "available_models": settings.get("OPENROUTER_MODELS", [
            "openai/gpt-4-turbo", "anthropic/claude-3-opus", "google/gemini-pro", 
            "meta-llama/llama-3-70b-instruct", "mistral/mistral-large"
        ]),
        "default_model": settings.get("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4-turbo"),
    },
    "deepseek": {
        "api_key": settings.get("DEEPSEEK_API_KEY", ""),
        "available_models": settings.get("DEEPSEEK_MODELS", ["deepseek-coder", "deepseek-chat"]),
        "default_model": settings.get("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
    },
    "google": {
        "api_key": settings.get("GOOGLE_API_KEY", ""),
        "available_models": settings.get("GOOGLE_MODELS", ["gemini-pro", "gemini-pro-vision", "gemini-ultra"]),
        "default_model": settings.get("GOOGLE_DEFAULT_MODEL", "gemini-pro"),
    },
    "xai": {
        "api_key": settings.get("XAI_API_KEY", ""),
        "available_models": settings.get("XAI_MODELS", ["xai-chat", "xai-instruct"]),
        "default_model": settings.get("XAI_DEFAULT_MODEL", "xai-chat"),
    },
    "qwen": {
        "api_key": settings.get("QWEN_API_KEY", ""),
        "available_models": settings.get("QWEN_MODELS", ["qwen-max", "qwen-plus", "qwen-turbo"]),
        "default_model": settings.get("QWEN_DEFAULT_MODEL", "qwen-max"),
    },
    "microsoft": {
        "api_key": settings.get("MICROSOFT_API_KEY", ""),
        "endpoint": settings.get("MICROSOFT_ENDPOINT", ""),
        "available_models": settings.get("MICROSOFT_MODELS", ["azure-gpt-4", "azure-gpt-35-turbo"]),
        "default_model": settings.get("MICROSOFT_DEFAULT_MODEL", "azure-gpt-4"),
    },
}

# Default AI provider
DEFAULT_PROVIDER = settings.get("DEFAULT_PROVIDER", "openai")

# Model registry for different task types
MODEL_REGISTRY = {
    "architect": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": settings.get("ARCHITECT_PROMPT", "You are an expert software architect specialized in designing scalable, maintainable systems."),
        "temperature": settings.get("ARCHITECT_TEMPERATURE", 0.3),
    },
    "implementer": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": settings.get("IMPLEMENTER_PROMPT", "You are an expert software developer with deep knowledge of best practices and patterns."),
        "temperature": settings.get("IMPLEMENTER_TEMPERATURE", 0.2),
    },
    "reviewer": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": settings.get("REVIEWER_PROMPT", "You are an expert code reviewer focused on code quality, security, and performance."),
        "temperature": settings.get("REVIEWER_TEMPERATURE", 0.3),
    },
    "integrator": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": settings.get("INTEGRATOR_PROMPT", "You are an integration specialist who excels at connecting different components and systems."),
        "temperature": settings.get("INTEGRATOR_TEMPERATURE", 0.4),
    },
}

# Database configuration
DATABASE_URL = settings.get("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR}/data/ptolemy.db")
