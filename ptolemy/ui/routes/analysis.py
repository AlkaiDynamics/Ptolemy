"""
PTOLEMY UI - Analysis routes for insights and feedback visualization.
"""
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from ptolemy.ui.app import templates, temporal_core, feedback_orchestrator

router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.get("/", response_class=HTMLResponse)
async def analysis_page(request: Request):
    """Render the analysis dashboard with feedback visualization."""
    # Get feedback events
    feedback_events = await temporal_core.get_events({"type": "feedback_recorded"})
    
    # Organize feedback by type
    feedback_by_type = {}
    for event in feedback_events:
        feedback_type = event["data"].get("feedback_type", "unknown")
        if feedback_type not in feedback_by_type:
            feedback_by_type[feedback_type] = []
        feedback_by_type[feedback_type].append(event)
    
    return templates.TemplateResponse(
        "analysis.html", 
        {
            "request": request,
            "feedback_by_type": feedback_by_type,
            "feedback_count": len(feedback_events)
        }
    )

@router.get("/timeline", response_class=HTMLResponse)
async def timeline_page(request: Request):
    """Render the timeline visualization page."""
    # Get all events
    events = await temporal_core.get_events({})
    
    # Group events by date
    events_by_date = {}
    for event in events:
        date = event["timestamp"].split("T")[0]
        if date not in events_by_date:
            events_by_date[date] = []
        events_by_date[date].append(event)
    
    return templates.TemplateResponse(
        "timeline.html", 
        {
            "request": request,
            "events_by_date": events_by_date,
            "event_count": len(events)
        }
    )

@router.post("/feedback/api/record")
async def record_feedback(
    feedback_type: str = Form(...),
    content: str = Form(...),
    source: str = Form("web_ui")
):
    """Record user feedback."""
    try:
        # Record feedback event
        event = await temporal_core.record_event("feedback_recorded", {
            "feedback_type": feedback_type,
            "content": content,
            "source": source
        })
        
        # Process feedback through the orchestrator
        await feedback_orchestrator.process_feedback(feedback_type, content)
        
        return JSONResponse({
            "success": True,
            "event_id": event["id"]
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
