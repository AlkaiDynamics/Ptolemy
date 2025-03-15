"""
PTOLEMY: AI-Accelerated Development System

This module provides the main entry point for the PTOLEMY system.
"""

import asyncio
import sys
from pathlib import Path

from loguru import logger

from ptolemy.config import logger
from ptolemy.temporal_core import TemporalCore
from ptolemy.context_engine import ContextEngine
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.feedback import FeedbackOrchestrator


async def initialize():
    """Initialize all PTOLEMY system components."""
    temporal_core = TemporalCore()
    context_engine = ContextEngine(temporal_core)
    multi_model = MultiModelProcessor(context_engine)
    feedback_orchestrator = FeedbackOrchestrator(temporal_core, context_engine)
    
    try:
        await temporal_core.initialize()
        await context_engine.initialize()
        logger.info("PTOLEMY system initialized successfully")
        return {
            "temporal_core": temporal_core,
            "context_engine": context_engine,
            "multi_model": multi_model,
            "feedback_orchestrator": feedback_orchestrator
        }
    except Exception as e:
        logger.error(f"Failed to initialize PTOLEMY system: {str(e)}")
        return None


def main():
    """Main entry point for the PTOLEMY system."""
    logger.info("Starting PTOLEMY system")
    components = asyncio.run(initialize())
    if components:
        logger.info("PTOLEMY system ready")
        return 0
    else:
        logger.error("PTOLEMY system failed to initialize")
        return 1


if __name__ == "__main__":
    sys.exit(main())
