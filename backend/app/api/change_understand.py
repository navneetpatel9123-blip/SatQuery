from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.schemas import AnalysisResult
from app.services.raster_ingestion_service import raster_ingestion_service
from app.services.change_pipeline import change_pipeline
# Assuming Phase 3 stores its results somewhere, for now we will just use a mock DB
# since the instructions said "input should reference an existing Phase 3 analysis/result"
# and we see analysis_db in app.api.analysis

from app.api.analysis import analysis_db

# We'll also need a place for Phase 3 AnalysisResults if analysis_db only has AnalysisResponse.
# But for now let's just make it possible to pass in change regions if they aren't in DB,
# or we'll fetch from a shared state.
# Actually, the user asked to "validate the Phase 3 result". 
# The Phase 3 AnalysisResult has `change_regions`. Let's mock a phase3_db if it doesn't exist, 
# or just allow passing it in for testing.

router = APIRouter()

# Global dict to store Phase 3 results if they aren't in analysis_db
phase3_db: Dict[str, AnalysisResult] = {}

class ChangeUnderstandRequest(BaseModel):
    analysis_id: Optional[str] = "auto"
    t1_image_id: str
    t2_image_id: str
    options: Optional[Dict[str, Any]] = None

@router.post("/understand")
async def understand_changes(request: ChangeUnderstandRequest):
    """Run the Phase 4 Change Understander on a Phase 3 analysis result or auto-detect changes."""
    # 1. Load T1/T2 inputs
    t1_asset = raster_ingestion_service.get_asset(request.t1_image_id)
    if not t1_asset:
        raise HTTPException(status_code=404, detail=f"T1 image {request.t1_image_id} not found.")
        
    t2_asset = raster_ingestion_service.get_asset(request.t2_image_id)
    if not t2_asset:
        raise HTTPException(status_code=404, detail=f"T2 image {request.t2_image_id} not found.")

    # 2. Get or auto-detect change regions
    regions = []
    if request.analysis_id and request.analysis_id in phase3_db:
        phase3_result = phase3_db[request.analysis_id]
        regions = getattr(phase3_result, "change_regions", [])
    elif request.analysis_id and request.analysis_id in analysis_db:
        phase3_result = analysis_db[request.analysis_id]
        regions = getattr(phase3_result, "change_regions", [])
    else:
        # Fallback to automatic spectral diff change detection
        from app.models.change_detector import SpectralDiffChangeDetector
        detector = SpectralDiffChangeDetector()
        detection_res = detector.detect_changes(t1_asset, t2_asset, options=request.options)
        regions = detection_res.regions


    # 3/4/5/6/7. Run Phase 4 pipeline and handle everything
    try:
        pipeline_result = change_pipeline.run_pipeline(
            t1_asset=t1_asset,
            t2_asset=t2_asset,
            change_regions=regions,
            options=request.options
        )
        return pipeline_result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evidence")
async def extract_evidence(request: ChangeUnderstandRequest):
    """Run the complete Phase 4 + Phase 5 pipeline to generate evidence-grounded results."""
    # The change_pipeline now natively executes the EvidenceExtractor at the end of Phase 4
    # and attaches the evidence to the final result, so we can reuse the same workflow.
    return await understand_changes(request)
