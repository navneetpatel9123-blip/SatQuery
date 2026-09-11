from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.schemas import RasterAsset, PairValidationResult
from app.services.raster_ingestion_service import raster_ingestion_service
from app.config import settings

router = APIRouter()


class PairValidationRequest(BaseModel):
    asset1_id: str
    asset2_id: str


class IngestionMessageResponse(BaseModel):
    message: str
    success: bool


@router.post("/upload", response_model=RasterAsset)
async def upload_raster(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload a single satellite raster (GeoTIFF, TIFF, PNG, JPEG)."""
    # 1. Save upload securely to disk
    asset_id, saved_path, original_filename = await raster_ingestion_service.save_upload(file)
    
    # 2. Extract metadata and run initial validation
    asset = raster_ingestion_service.process_and_validate(asset_id, saved_path, original_filename)
    
    # 3. Generate preview (PNG thumbnail) in the background if validation succeeded
    if asset.validation_status != "FAIL":
        background_tasks.add_task(raster_ingestion_service.generate_preview, asset_id)
        
    return asset


@router.post("/validate", response_model=RasterAsset)
async def validate_raster(
    image_id: str = Body(..., embed=True)
):
    """Re-validate an existing uploaded raster asset by its ID."""
    asset = raster_ingestion_service.get_asset(image_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found.")
        
    # Re-run metadata and validation
    file_path = settings.base_dir / asset.path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Raster file not found on disk at {asset.path}.")
        
    revalidated_asset = raster_ingestion_service.process_and_validate(image_id, file_path, asset.filename)
    return revalidated_asset


@router.post("/validate-pair", response_model=PairValidationResult)
async def validate_raster_pair(
    request: PairValidationRequest
):
    """Validate cross-modal pair (Optical + SAR) or bi-temporal pair (T1 + T2) compatibility."""
    return raster_ingestion_service.validate_pair(request.asset1_id, request.asset2_id)


@router.get("/{image_id}/metadata", response_model=RasterAsset)
async def get_raster_metadata(image_id: str):
    """Retrieve metadata and validation status for a specific image asset."""
    asset = raster_ingestion_service.get_asset(image_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found.")
    return asset


@router.get("/{image_id}/preview")
async def get_raster_preview(image_id: str):
    """Get the downsampled preview PNG for an image asset."""
    asset = raster_ingestion_service.get_asset(image_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found.")
        
    preview_path = settings.thumbnails_dir / f"{image_id}.png"
    if not preview_path.exists():
        # Try generating synchronously
        try:
            raster_ingestion_service.generate_preview(image_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Preview thumbnail is still generating or failed to generate.")
            
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview thumbnail not found.")
        
    return FileResponse(preview_path, media_type="image/png")


@router.delete("/{image_id}", response_model=IngestionMessageResponse)
async def delete_raster(image_id: str):
    """Delete an uploaded image asset and its associated files."""
    deleted = raster_ingestion_service.delete_asset(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Asset not found or already deleted.")
    return IngestionMessageResponse(message=f"Asset {image_id} deleted successfully.", success=True)


@router.get("", response_model=List[RasterAsset])
async def list_raster_assets():
    """Retrieve list of all uploaded and parsed raster assets."""
    return raster_ingestion_service.get_all_assets()


class DemoPairRequest(BaseModel):
    sample_name: Optional[str] = "test_102_0512_0000"


class DemoPairResponse(BaseModel):
    t1_asset: RasterAsset
    t2_asset: RasterAsset
    pair_validation: PairValidationResult
    dataset_name: str
    sample_id: str


@router.post("/demo-pair", response_model=DemoPairResponse)
async def load_demo_pair(request: DemoPairRequest):
    """Load authentic LEVIR-CD benchmark demo pair into active workspace."""
    sample_name = request.sample_name or "test_102_0512_0000"
    try:
        t1, t2 = raster_ingestion_service.load_demo_pair(sample_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    pair_val = raster_ingestion_service.validate_pair(t1.id, t2.id)
    return DemoPairResponse(
        t1_asset=t1,
        t2_asset=t2,
        pair_validation=pair_val,
        dataset_name="LEVIR-CD",
        sample_id=sample_name,
    )


