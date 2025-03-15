#!/usr/bin/env python
import asyncio
import json
import os
import sys
from pathlib import Path

import click
from loguru import logger

from ptolemy.config import logger
from ptolemy.temporal_core import TemporalCore
from ptolemy.context_engine import ContextEngine
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.feedback import FeedbackOrchestrator


# Initialize core components
temporal_core = TemporalCore()
context_engine = ContextEngine(temporal_core)
multi_model = MultiModelProcessor(context_engine)
feedback_orchestrator = FeedbackOrchestrator(temporal_core, context_engine)


async def initialize_system():
    """Initialize all PTOLEMY system components."""
    try:
        await temporal_core.initialize()
        await context_engine.initialize()
        logger.info("PTOLEMY system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PTOLEMY system: {str(e)}")
        return False


@click.group()
def cli():
    """PTOLEMY AI-Accelerated Development System."""
    pass


@cli.command()
@click.option('--name', prompt='Project name', default='my-ptolemy-project', help='Name of the project')
def init(name):
    """Initialize a new PTOLEMY project."""
    async def _init():
        initialized = await initialize_system()
        if initialized:
            await temporal_core.record_event("project_initialized", {"name": name})
            logger.info(f"Project '{name}' initialized successfully")
            click.echo(f"Project '{name}' initialized successfully")
        else:
            click.echo("Failed to initialize project. Check logs for details.")
    
    asyncio.run(_init())


@cli.command()
@click.argument('prompt')
@click.option('--model', '-m', default='implementer', help='Model to use (architect, implementer, reviewer, integrator)')
def generate(prompt, model):
    """Generate code or content from a prompt."""
    async def _generate():
        await initialize_system()
        try:
            result = await multi_model.route_task(prompt, model)
            click.echo("\nGenerated output:")
            click.echo("=================")
            click.echo(result)
            
            # Record this as an event
            await temporal_core.record_event("content_generated", {
                "prompt": prompt,
                "model": model,
                "result_length": len(result)
            })
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            click.echo(f"Generation failed: {str(e)}")
    
    asyncio.run(_generate())


@cli.command()
@click.argument('initial_prompt')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to chain configuration file')
def chain(initial_prompt, config):
    """Run a multi-stage prompt chain."""
    async def _chain():
        await initialize_system()
        try:
            # Default chain if no config provided
            stages = [
                {"name": "architecture", "model_type": "architect", "next_prompt": "Implement this architecture"},
                {"name": "implementation", "model_type": "implementer", "next_prompt": "Review this implementation"},
                {"name": "review", "model_type": "reviewer"}
            ]
            
            # Load stages from config file if provided
            if config:
                with open(config, 'r') as f:
                    config_data = json.load(f)
                    if "stages" in config_data:
                        stages = config_data["stages"]
            
            results = await multi_model.route_multi_stage(initial_prompt, stages)
            
            click.echo("\nChain Results:")
            click.echo("==============")
            for result in results:
                click.echo(f"\n## Stage: {result['stage']}")
                click.echo(result["output"])
            
            # Record this as an event
            await temporal_core.record_event("chain_executed", {
                "initial_prompt": initial_prompt,
                "stages": [s["name"] for s in stages],
                "result_count": len(results)
            })
        except Exception as e:
            logger.error(f"Chain execution failed: {str(e)}")
            click.echo(f"Chain execution failed: {str(e)}")
    
    asyncio.run(_chain())


@cli.command()
@click.argument('feedback_text')
@click.option('--type', '-t', default='user_preference', help='Type of feedback')
@click.option('--target', help='Target ID (e.g., event ID)')
def feedback(feedback_text, type, target):
    """Provide feedback to the system."""
    async def _feedback():
        await initialize_system()
        try:
            event = await feedback_orchestrator.record_feedback(
                type,
                feedback_text,
                "user",
                target
            )
            click.echo(f"Feedback recorded with ID: {event['id']}")
        except Exception as e:
            logger.error(f"Failed to record feedback: {str(e)}")
            click.echo(f"Failed to record feedback: {str(e)}")
    
    asyncio.run(_feedback())


@cli.command()
@click.option('--type', '-t', help='Filter by feedback type')
def analyze(type):
    """Analyze feedback trends."""
    async def _analyze():
        await initialize_system()
        try:
            analysis = await feedback_orchestrator.analyze_feedback_trends(type)
            click.echo("\nFeedback Analysis:")
            click.echo("=================")
            click.echo(f"Total feedback: {analysis['total_feedback']}")
            
            click.echo("\nBy Type:")
            for fb_type, count in analysis['by_type'].items():
                click.echo(f"- {fb_type}: {count}")
            
            click.echo("\nBy Source:")
            for source, count in analysis['by_source'].items():
                click.echo(f"- {source}: {count}")
            
            click.echo("\nRecent Feedback:")
            for fb in analysis['recent_feedback']:
                click.echo(f"- [{fb['timestamp']}] {fb['data']['feedback_type']}: {fb['data']['content']}")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            click.echo(f"Analysis failed: {str(e)}")
    
    asyncio.run(_analyze())


if __name__ == '__main__':
    cli()
