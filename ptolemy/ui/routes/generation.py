"""
PTOLEMY UI - Generation routes for AI content generation.
"""
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from ptolemy.ui.app import templates, multi_model, temporal_core

router = APIRouter(prefix="/generate", tags=["generation"])

@router.get("/", response_class=HTMLResponse)
async def generate_page(request: Request):
    """Render the generation interface page."""
    return templates.TemplateResponse(
        "generation.html", 
        {"request": request}
    )

@router.post("/api/completion")
async def generate_completion(
    prompt: str = Form(...),
    model_type: str = Form("implementer")
):
    """Generate content using the specified model type."""
    try:
        # Route the task to the appropriate model
        result = await multi_model.route_task(prompt, model_type)
        
        # Record the event
        await temporal_core.record_event("ui_generation", {
            "prompt": prompt,
            "model_type": model_type,
            "result_length": len(result)
        })
        
        return JSONResponse({
            "success": True,
            "result": result
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
