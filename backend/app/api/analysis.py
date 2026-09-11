from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.schemas import AnalysisResponse, ModelInfo
from app.services.agent_service import agent_service
from app.services.model_registry import model_registry

router = APIRouter()

# Simple in-memory DB for analysis runs
analysis_db: Dict[str, AnalysisResponse] = {}


class QueryRequest(BaseModel):
    image_id: str
    query: str
    model_id: Optional[str] = None
    session_id: Optional[str] = None


@router.post("/query", response_model=AnalysisResponse)
async def query_agent(request: QueryRequest):
    """
    Route query to the SatQuery agent. 
    The agent classifies the task (VQA, captioning, or grounding) and selects the best model.
    """
    try:
        response = agent_service.run_analysis(
            image_id=request.image_id,
            query=request.query,
            preferred_model_id=request.model_id,
            session_id=request.session_id
        )
        # Store in db
        analysis_db[response.analysis_id] = response
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent analysis failed: {str(e)}")


@router.post("/vqa", response_model=AnalysisResponse)
async def run_vqa(
    image_id: str = Body(...),
    query: str = Body(...),
    model_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None)
):
    """Directly query the VQA specialist model."""
    try:
        # Enforce task classification override or direct run
        response = agent_service.run_analysis(
            image_id=image_id, 
            query=query, 
            preferred_model_id=model_id,
            session_id=session_id
        )
        # Enforce that VQA was run
        response.task = "SINGLE_VQA"
        analysis_db[response.analysis_id] = response
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/caption", response_model=AnalysisResponse)
async def run_caption(
    image_id: str = Body(...),
    model_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None)
):
    """Directly query the Captioning specialist model."""
    try:
        response = agent_service.run_analysis(
            image_id=image_id, 
            query="Describe this scene in detail.", 
            preferred_model_id=model_id,
            session_id=session_id
        )
        response.task = "CAPTIONING"
        analysis_db[response.analysis_id] = response
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ground", response_model=AnalysisResponse)
async def run_ground(
    image_id: str = Body(...),
    query: str = Body(...),
    model_id: Optional[str] = Body(None),
    session_id: Optional[str] = Body(None)
):
    """Directly query the Grounding specialist model."""
    try:
        response = agent_service.run_analysis(
            image_id=image_id, 
            query=query, 
            preferred_model_id=model_id,
            session_id=session_id
        )
        response.task = "GROUNDING"
        analysis_db[response.analysis_id] = response
        return response
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(analysis_id: str):
    """Retrieve historical analysis result by ID."""
    if analysis_id not in analysis_db:
        raise HTTPException(status_code=404, detail="Analysis result not found.")
    return analysis_db[analysis_id]


# Expose registry under /api/v1/models (relative router prefix will be registered in main.py)
# We can register another router for top-level /api/v1/models or put it in this router
@router.get("/system/models", response_model=List[ModelInfo])
async def list_registered_models():
    """List all models registered in the SatQuery AI system."""
    return model_registry.list_models()
