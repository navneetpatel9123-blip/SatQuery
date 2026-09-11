"""
Health check endpoint for SatQuery AI backend.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    capabilities: list[str]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    return HealthResponse(
        status="operational",
        service="SatQuery AI",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        capabilities=[
            "geotiff_ingestion",
            "change_detection",
            "change_understanding",
            "evidence_extraction",
            "execution_trace",
            "vqa",
        ],
    )
