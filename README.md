# PTOLEMY: AI-Accelerated Development System

PTOLEMY (Prompt Trees Orchestrating Language-Model Enhanced Modular Y-Trees) leverages advanced AI systems to convert conversationally described requirements into immediate, high-quality software implementations.

## Overview

PTOLEMY introduces an AI-driven software development methodology through:

- AI-generated incremental improvements
- End-to-end AI-based design and implementation
- Continuous, iterative generation processes
- Human oversight at strategic checkpoints for validation
- Strategic "Vibe Coding" sessions guiding high-level direction
- Recursive self-enhancement using emergent AI capabilities

## Project Structure

```
ptolemy/
├── data/               # Data storage
│   ├── context/        # Context engine data
│   └── temporal/       # Temporal core data
├── logs/               # Log files
├── ptolemy/            # Source code
│   ├── __init__.py     # Package initialization
│   ├── config.py       # Configuration module
│   ├── context_engine.py # Context engine components
│   ├── database.py     # Database models and operations
│   ├── feedback.py     # Feedback orchestrator
│   ├── multi_model.py  # Multi-model processor
│   ├── temporal_core.py # Temporal core components
│   ├── ui/             # Web UI components
│   └── utils.py        # Utility functions
├── templates/          # Template files
├── tests/              # Test files
├── venv/               # Virtual environment
├── .env                # Environment variables
├── cli.py              # Command-line interface
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Getting Started

1. Clone this repository
2. Set up the Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Configure your `.env` file with your OpenAI API key
4. Initialize the project:
   ```bash
   python cli.py init
   ```

## Environment Setup

Make sure to set up your environment variables in the `.env` file:

```
OPENAI_API_KEY=your_api_key_here
DEFAULT_MODEL=gpt-4
LOG_LEVEL=info
```

## CLI Usage

The PTOLEMY system provides a command-line interface for easy interaction:

```bash
# Initialize a new project
python cli.py init

# Generate content from a prompt
python cli.py generate "Create a Python function to calculate Fibonacci numbers"

# Run a multi-stage prompt chain
python cli.py chain "Design a REST API for a blog system"

# Provide feedback to the system
python cli.py feedback "I prefer more detailed comments in the code" --type user_preference

# Analyze feedback trends
python cli.py analyze
```

## Core Components

### Temporal Core
- Continuous event-stream management replacing linear version control
- Comprehensive project history and decision rationale logging

### Context Engine
- Relationship mapping between project entities
- Repository for reusable implementation patterns
- Management of project-specific insights

### Multi-Model Processor
- Task delegation to specialized AI models
- Integration of independently generated components
- Ensuring consistency across software modules

### Feedback Orchestrator
- Analysis of user-driven changes
- Continuous adjustment of AI-generation quality
- Adaptive responses to project needs and user preferences

## Key Components

### Temporal Core
Manages event recording, retrieval, and temporal analysis of development activities.

### Context Engine
Maintains contextual understanding of the project, including patterns, relationships, and insights.

### Multi-Model Processor
Orchestrates AI model interactions, delegating tasks to specialized models and integrating their outputs.

### Database Integration
Uses SQLAlchemy with SQLite for persistent storage of events, relationships, patterns, and insights.

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## License

ISC
