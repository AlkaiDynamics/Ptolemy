"""
PTOLEMY UI - Project routes for managing and visualizing projects.
"""
from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
from pathlib import Path
from datetime import datetime

from ptolemy.ui.app import templates, temporal_core

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("/", response_class=HTMLResponse)
async def projects_page(request: Request):
    """Render the projects dashboard with celestial visualization."""
    # Get projects from events
    events = await temporal_core.get_events({"type": "project_initialized"})
    
    projects = []
    for event in events:
        project_id = event["id"]
        project_name = event["data"].get("name", "Unnamed Project")
        
        # Count events related to this project
        project_events = await temporal_core.get_events({"project_id": project_id})
        
        projects.append({
            "id": project_id,
            "name": project_name,
            "description": event["data"].get("description", ""),
            "created": datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d"),
            "event_count": len(project_events)
        })
    
    return templates.TemplateResponse(
        "projects.html", 
        {
            "request": request,
            "projects": projects,
            "projects_json": json.dumps(projects)
        }
    )

@router.get("/new", response_class=HTMLResponse)
async def new_project_page(request: Request):
    """Render the new project creation page."""
    return templates.TemplateResponse(
        "new_project.html", 
        {"request": request}
    )

@router.post("/new")
async def create_project(
    name: str = Form(...),
    description: str = Form("")
):
    """Create a new project and redirect to its details page."""
    event = await temporal_core.record_event("project_initialized", {
        "name": name,
        "description": description
    })
    
    return RedirectResponse(f"/projects/{event['id']}", status_code=303)

@router.get("/{project_id}", response_class=HTMLResponse)
async def project_details(request: Request, project_id: str):
    """Render the project details page with timeline visualization."""
    # Get project info
    try:
        event = await temporal_core.get_event_by_id(project_id)
        if event["type"] != "project_initialized":
            raise HTTPException(status_code=404, detail="Project not found")
    except Exception:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get project events
    project_events = await temporal_core.get_events({"project_id": project_id})
    
    return templates.TemplateResponse(
        "project_details.html", 
        {
            "request": request,
            "project": {
                "id": project_id,
                "name": event["data"].get("name", "Unnamed Project"),
                "description": event["data"].get("description", ""),
                "created": datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d %H:%M")
            },
            "events": project_events
        }
    )
