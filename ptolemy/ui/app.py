"""
PTOLEMY UI - FastAPI application for the celestial-themed web interface.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

# Import PTOLEMY components
from ptolemy.temporal_core import TemporalCore
from ptolemy.context_engine import ContextEngine
from ptolemy.multi_model import MultiModelProcessor
from ptolemy.feedback import FeedbackOrchestrator

# Initialize app
app = FastAPI(title="PTOLEMY")

# Configure templates and static files
templates_path = Path(__file__).parent / "templates"
static_path = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=templates_path)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Initialize PTOLEMY components (shared across requests)
temporal_core = TemporalCore()
context_engine = ContextEngine()
multi_model = MultiModelProcessor(context_engine)
feedback_orchestrator = FeedbackOrchestrator()

@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    # Initialize components that require async initialization
    pass

@app.get("/")
async def index(request: Request):
    """Render the index page."""
    return templates.TemplateResponse("index.html", {"request": request})

# Include routers
# These imports are placed here to avoid circular imports
from ptolemy.ui.routes import projects, generation, analysis

# Register routers
app.include_router(projects.router)
app.include_router(generation.router)
app.include_router(analysis.router)
