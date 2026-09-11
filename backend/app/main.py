"""
SatQuery AI Backend
Bi-Temporal Satellite Image Change Detection & VQA API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from datetime import datetime, timezone

from app.api import health
from app.api.ingestion import router as ingestion_router
from app.api.analysis import router as analysis_router
from app.api.change_understand import router as change_understand_router
from app.api.orchestrate import router as orchestrate_router
from app.services.model_registry import model_registry
from app.schemas import ModelInfo
from typing import List

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    print("[SatQuery AI] Backend starting...")
    print(f"[SatQuery AI] Time: {datetime.now(timezone.utc).isoformat()}")
    print("[SatQuery AI] Status: Ready")
    yield

app = FastAPI(
    title="SatQuery AI",
    description="Bi-Temporal Satellite Image Change Detection & Visual Question Answering API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["System"])
app.include_router(health.router, prefix="/api/v1", tags=["System"])
app.include_router(ingestion_router, prefix="/api/v1/ingestion", tags=["Raster Ingestion"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Raster Analysis"])
app.include_router(change_understand_router, prefix="/api/v1/change", tags=["Change Understanding"])
app.include_router(orchestrate_router, prefix="/api/v1", tags=["Orchestration"])


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root path to interactive FastAPI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api/v1/models", response_model=List[ModelInfo], tags=["Model Registry"])
async def get_system_models():
    """List all models registered in the SatQuery AI system."""
    return model_registry.list_models()

