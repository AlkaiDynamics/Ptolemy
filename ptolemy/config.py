import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys

# Load environment variables from .env file
load_dotenv()

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

# AI Provider Configurations
AI_PROVIDERS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "available_models": [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        ],
        "default_model": os.getenv("DEFAULT_MODEL", "gpt-4"),
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "available_models": [
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"
        ],
        "default_model": os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229"),
    },
    "mistral": {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "available_models": [
            "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large-latest"
        ],
        "default_model": os.getenv("MISTRAL_DEFAULT_MODEL", "mistral-medium"),
    },
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "available_models": [
            "llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"
        ],
        "default_model": os.getenv("GROQ_DEFAULT_MODEL", "mixtral-8x7b-32768"),
    },
    "cohere": {
        "api_key": os.getenv("COHERE_API_KEY"),
        "available_models": [
            "command", "command-light", "command-nightly", "command-r", "command-r-plus"
        ],
        "default_model": os.getenv("COHERE_DEFAULT_MODEL", "command-r"),
    },
    "ollama": {
        "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        "available_models": [
            "llama2", "llama2:13b", "llama2:70b", "mistral", "mixtral", "phi", "gemma", "codellama"
        ],
        "default_model": os.getenv("OLLAMA_DEFAULT_MODEL", "mixtral"),
    },
    "lmstudio": {
        "api_base": os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1"),
        "available_models": ["local-model"],
        "default_model": os.getenv("LMSTUDIO_DEFAULT_MODEL", "local-model"),
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "available_models": [
            "openai/gpt-4-turbo", "anthropic/claude-3-opus", "google/gemini-pro", 
            "meta-llama/llama-3-70b-instruct", "mistral/mistral-large"
        ],
        "default_model": os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4-turbo"),
    },
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "available_models": ["deepseek-coder", "deepseek-chat"],
        "default_model": os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
    },
    "google": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "available_models": ["gemini-pro", "gemini-pro-vision", "gemini-ultra"],
        "default_model": os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-pro"),
    },
    "xai": {
        "api_key": os.getenv("XAI_API_KEY"),
        "available_models": ["xai-chat", "xai-instruct"],
        "default_model": os.getenv("XAI_DEFAULT_MODEL", "xai-chat"),
    },
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY"),
        "available_models": ["qwen-max", "qwen-plus", "qwen-turbo"],
        "default_model": os.getenv("QWEN_DEFAULT_MODEL", "qwen-max"),
    },
    "microsoft": {
        "api_key": os.getenv("MICROSOFT_API_KEY"),
        "endpoint": os.getenv("MICROSOFT_ENDPOINT"),
        "available_models": ["azure-gpt-4", "azure-gpt-35-turbo"],
        "default_model": os.getenv("MICROSOFT_DEFAULT_MODEL", "azure-gpt-4"),
    },
}

# Default AI provider
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")

# Model registry for different task types
MODEL_REGISTRY = {
    "architect": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": "You are an expert software architect specialized in designing scalable, maintainable systems.",
        "temperature": 0.3,
    },
    "implementer": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": "You are an expert software developer with deep knowledge of best practices and patterns.",
        "temperature": 0.2,
    },
    "reviewer": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": "You are an expert code reviewer focused on code quality, security, and performance.",
        "temperature": 0.3,
    },
    "integrator": {
        "provider": DEFAULT_PROVIDER,
        "model": AI_PROVIDERS[DEFAULT_PROVIDER]["default_model"],
        "system_prompt": "You are an integration specialist who excels at connecting different components and systems.",
        "temperature": 0.4,
    },
}
