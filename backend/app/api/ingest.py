import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from datetime import datetime
import shutil

from app.config import settings
from app.schemas import GeoTIFFMetadata, ImagePairValidation
from app.services.geotiff_service import geotiff_service
from pydantic import BaseModel

router = APIRouter()

class UploadPairResponse(BaseModel):
    pair_id: str
    message: str
    validation: ImagePairValidation

# Simple in-memory store for demo. In production, use DB.
pairs_db = {}


@router.post("/images/upload", response_model=UploadPairResponse)
async def upload_images(
    t1_file: UploadFile = File(...),
    t2_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload a pair of GeoTIFF images (T1 and T2)."""
    
    if not t1_file.filename.endswith(('.tif', '.tiff')) or not t2_file.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files (.tif, .tiff) are supported.")
        
    pair_id = str(uuid.uuid4())
    pair_dir = settings.upload_dir / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    t1_path = pair_dir / "t1.tif"
    t2_path = pair_dir / "t2.tif"
    
    # Save files
    with open(t1_path, "wb") as buffer:
        shutil.copyfileobj(t1_file.file, buffer)
    with open(t2_path, "wb") as buffer:
        shutil.copyfileobj(t2_file.file, buffer)
        
    # Validate
    try:
        validation = geotiff_service.validate_pair(t1_path, t2_path)
    except Exception as e:
        # Clean up on validation failure reading
        shutil.rmtree(pair_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to read GeoTIFF metadata: {str(e)}")

    if not validation.is_valid:
        # We might want to keep it to let user see why it failed, but let's keep it simple for now
        pass 
        
    # Schedule thumbnail generation
    t1_thumb = settings.thumbnails_dir / f"{pair_id}_t1.png"
    t2_thumb = settings.thumbnails_dir / f"{pair_id}_t2.png"
    
    background_tasks.add_task(geotiff_service.generate_thumbnail, t1_path, t1_thumb)
    background_tasks.add_task(geotiff_service.generate_thumbnail, t2_path, t2_thumb)
    
    # Save to db
    pairs_db[pair_id] = {
        "t1_path": str(t1_path),
        "t2_path": str(t2_path),
        "validation": validation.model_dump()
    }
    
    return UploadPairResponse(
        pair_id=pair_id,
        message="Upload successful",
        validation=validation
    )


@router.get("/images/{pair_id}")
async def get_pair_info(pair_id: str):
    """Get metadata and validation status for a pair."""
    if pair_id not in pairs_db:
        raise HTTPException(status_code=404, detail="Pair not found")
        
    t1_path = Path(pairs_db[pair_id]["t1_path"])
    t2_path = Path(pairs_db[pair_id]["t2_path"])
    
    t1_meta = geotiff_service.extract_metadata(t1_path)
    t2_meta = geotiff_service.extract_metadata(t2_path)
    
    return {
        "pair_id": pair_id,
        "t1_metadata": t1_meta,
        "t2_metadata": t2_meta,
        "validation": pairs_db[pair_id]["validation"]
    }


@router.get("/images/{pair_id}/preview/{time_step}")
async def get_preview(pair_id: str, time_step: str):
    """Get thumbnail preview for T1 or T2."""
    if time_step not in ["t1", "t2"]:
        raise HTTPException(status_code=400, detail="time_step must be 't1' or 't2'")
        
    thumb_path = settings.thumbnails_dir / f"{pair_id}_{time_step}.png"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found or still generating")
        
    return FileResponse(thumb_path, media_type="image/png")
