import os
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import RasterioIOError
from fastapi import UploadFile, HTTPException

from app.config import settings
from app.schemas import RasterAsset, PairValidationResult


class RasterIngestionService:
    """Service to handle GeoTIFF, TIFF, and alternative image uploads, parsing, metadata extraction, validation, and preview generation."""

    def __init__(self):
        # We ensure settings directories are created
        settings.ensure_dirs()

    def get_all_assets(self) -> List[RasterAsset]:
        """Load and return all stored assets from files."""
        assets = []
        if not settings.upload_dir.exists():
            return assets
            
        for path in settings.upload_dir.iterdir():
            if path.is_dir():
                json_path = path / "asset.json"
                if json_path.exists():
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                            # Convert timestamp back to datetime
                            if data.get("timestamp"):
                                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                            assets.append(RasterAsset(**data))
                    except Exception:
                        # Skip malformed json files
                        pass
        return assets

    def get_asset(self, asset_id: str) -> Optional[RasterAsset]:
        """Load and return a single asset by its ID."""
        json_path = settings.upload_dir / asset_id / "asset.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if data.get("timestamp"):
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                return RasterAsset(**data)
        except Exception:
            return None

    def delete_asset(self, asset_id: str) -> bool:
        """Delete asset files and metadata."""
        asset_dir = settings.upload_dir / asset_id
        thumbnail_path = settings.thumbnails_dir / f"{asset_id}.png"
        
        deleted = False
        if asset_dir.exists():
            shutil.rmtree(asset_dir, ignore_errors=True)
            deleted = True
        if thumbnail_path.exists():
            try:
                os.remove(thumbnail_path)
            except OSError:
                pass
        return deleted

    async def save_upload(self, upload_file: UploadFile) -> Tuple[str, Path, str]:
        """Securely store an uploaded file to a temporary location, then move it to asset dir."""
        asset_id = str(uuid.uuid4())
        asset_dir = settings.upload_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename (remove path traversal attempts)
        original_filename = Path(upload_file.filename).name
        saved_path = asset_dir / original_filename
        
        # Write chunks to prevent memory bloat
        try:
            with open(saved_path, "wb") as buffer:
                while chunk := await upload_file.read(1024 * 1024):
                    buffer.write(chunk)
        except Exception as e:
            shutil.rmtree(asset_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Failed to write file to disk: {str(e)}")
            
        return asset_id, saved_path, original_filename

    def process_and_validate(self, asset_id: str, file_path: Path, filename: str) -> RasterAsset:
        """Validate raster structure, extract metadata, and save asset description."""
        errors: List[str] = []
        warnings: List[str] = []
        validation_status = "PASS"
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # 1. Size Validation
        if file_size_mb > settings.max_file_size_mb:
            errors.append(f"Upload size limit violation: {file_size_mb:.1f}MB exceeds the maximum of {settings.max_file_size_mb}MB.")
            validation_status = "FAIL"
            
        # 2. Extension Validation
        ext = file_path.suffix.lower()
        allowed_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        if ext not in allowed_exts:
            errors.append(f"Unsupported file extension: {ext}. Only GeoTIFF, TIFF, PNG, and JPEG are supported.")
            validation_status = "FAIL"
            
        # 3. Readability & Structure Validation
        width = 0
        height = 0
        bands = 0
        dtype = "unknown"
        crs = "UNKNOWN"
        epsg = None
        bounds = [0.0, 0.0, 0.0, 0.0]
        resolution = [0.0, 0.0]
        transform = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        nodata = None
        timestamp = None
        modality = "unknown"
        modality_certain = False
        
        if validation_status != "FAIL":
            try:
                with rasterio.open(file_path) as src:
                    width = src.width
                    height = src.height
                    bands = src.count
                    dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
                    nodata = src.nodata
                    
                    # Coordinate Reference System
                    if src.crs:
                        crs = src.crs.to_string()
                        epsg = src.crs.to_epsg()
                    else:
                        warnings.append("Image loaded successfully, but no coordinate reference system was found. Geospatial pair analysis may be limited.")
                        validation_status = "WARNING"
                        
                    # Bounds & Resolution
                    if src.bounds:
                        bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                    else:
                        warnings.append("Geospatial bounds are missing or invalid.")
                        validation_status = "WARNING"
                        
                    if hasattr(src, "res") and src.res:
                        resolution = list(src.res)
                    else:
                        resolution = [0.0, 0.0]
                        
                    if src.transform:
                        transform = [
                            src.transform.a, src.transform.b, src.transform.c,
                            src.transform.d, src.transform.e, src.transform.f
                        ]
                        # Check for identity transform (typical of non-georeferenced images like PNG/JPEG)
                        if transform == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] or transform == [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]:
                            warnings.append("Image lacks georeferencing information. Coordinate transform is default identity.")
                            if validation_status == "PASS":
                                validation_status = "WARNING"
                    
                    # Metadata timestamp extraction from tags
                    tags = src.tags()
                    if "TIFFTAG_DATETIME" in tags:
                        try:
                            # Standard TIFF tag datetime format: YYYY:MM:DD HH:MM:SS
                            timestamp = datetime.strptime(tags["TIFFTAG_DATETIME"], "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass
                            
                    # Modality Detection
                    # Look at keywords first
                    fn_lower = filename.lower()
                    sar_keywords = ["sar", "s1", "sentinel-1", "sentinel1", "vv", "vh", "polarization", "amplitude", "intensity"]
                    has_sar_keyword = any(k in fn_lower for k in sar_keywords)
                    
                    if has_sar_keyword:
                        modality = "sar"
                        modality_certain = True
                    elif bands == 1:
                        modality = "sar"
                        modality_certain = False
                        warnings.append("Single-band raster detected. Assuming SAR/Grayscale modality (uncertain).")
                        if validation_status == "PASS":
                            validation_status = "WARNING"
                    elif bands == 3 or bands == 4:
                        modality = "optical"
                        modality_certain = True
                    elif bands > 4:
                        modality = "multispectral"
                        modality_certain = True
                    else:
                        modality = "unknown"
                        modality_certain = False
                        warnings.append("Could not reliably determine image modality.")
                        if validation_status == "PASS":
                            validation_status = "WARNING"
                            
            except RasterioIOError as rio_err:
                errors.append(f"Unable to read this raster. The file may be corrupted or unsupported. (Details: {str(rio_err)})")
                validation_status = "FAIL"
            except Exception as exc:
                errors.append(f"Unexpected error parsing raster structure: {str(exc)}")
                validation_status = "FAIL"
                
        # Generate the standardized asset object
        asset = RasterAsset(
            id=asset_id,
            filename=filename,
            path=str(file_path.relative_to(settings.base_dir).as_posix()) if file_path.is_relative_to(settings.base_dir) else str(file_path.resolve().as_posix()),
            modality=modality,
            modality_certain=modality_certain,
            width=width,
            height=height,
            bands=bands,
            dtype=dtype,
            crs=crs,
            epsg=epsg,
            bounds=bounds,
            resolution=resolution,
            transform=transform,
            nodata=nodata,
            timestamp=timestamp,
            validation_status=validation_status,
            warnings=warnings,
            errors=errors
        )
        
        # Save JSON representation next to file
        self.save_asset_json(asset)
        
        return asset

    def save_asset_json(self, asset: RasterAsset):
        """Save asset Pydantic model to asset.json next to original file."""
        asset_dir = settings.upload_dir / asset.id
        asset_dir.mkdir(parents=True, exist_ok=True)
        json_path = asset_dir / "asset.json"
        
        # Custom dict serializing datetime to string
        asset_dict = asset.model_dump()
        if asset_dict.get("timestamp"):
            asset_dict["timestamp"] = asset_dict["timestamp"].isoformat()
            
        with open(json_path, "w") as f:
            json.dump(asset_dict, f, indent=2)

    def generate_preview(self, asset_id: str) -> Path:
        """Generate a downsampled preview PNG preserving aspect ratio and write to data/thumbnails/{id}.png."""
        asset = self.get_asset(asset_id)
        if not asset or asset.validation_status == "FAIL":
            raise FileNotFoundError(f"Valid asset {asset_id} not found.")
            
        file_path = Path(asset.path) if Path(asset.path).is_absolute() else settings.base_dir / asset.path
        if not file_path.exists():
            raise FileNotFoundError(f"Asset file {file_path} does not exist.")
            
        output_path = settings.thumbnails_dir / f"{asset_id}.png"
        
        try:
            with rasterio.open(file_path) as src:
                w, h = src.width, src.height
                max_dim = settings.thumbnail_size[0]
                
                # Calculate aspect-ratio downsampled sizes
                if w > h:
                    new_w = max_dim
                    new_h = int(h * (max_dim / w))
                else:
                    new_h = max_dim
                    new_w = int(w * (max_dim / h))
                
                # Prevent dimensions being 0
                new_w = max(1, new_w)
                new_h = max(1, new_h)
                
                # Read band selection
                # Grayscale for SAR / single-band, RGB for optical / multispectral
                num_bands = min(src.count, 3)
                band_indices = list(range(1, num_bands + 1))
                
                # Windowed downsampling during read to avoid loading whole image to RAM
                out_shape = (num_bands, new_h, new_w)
                img_array = src.read(
                    indexes=band_indices,
                    out_shape=out_shape,
                    resampling=rasterio.enums.Resampling.bilinear
                )
                
                # Normalize values to 0-255 uint8 range safely
                # Mask out nodata values if they are defined
                mask = np.ones_like(img_array, dtype=bool)
                if src.nodata is not None:
                    mask = (img_array != src.nodata)
                    
                # Safe percentiles for stretching contrast (ignore outliers)
                if np.any(mask):
                    valid_vals = img_array[mask]
                    p2, p98 = np.percentile(valid_vals, (2, 98))
                else:
                    p2, p98 = 0, 1
                    
                if p2 == p98:
                    p98 = p2 + 1.0  # Avoid division by zero
                    
                img_array = img_array.astype(np.float32)
                normalized = np.clip((img_array - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                
                # Handle shapes for PIL (H, W, C)
                if num_bands == 1:
                    pil_img = Image.fromarray(normalized[0])
                elif num_bands == 2:
                    # Construct pseudo-RGB
                    r = normalized[0]
                    g = normalized[1]
                    b = np.zeros_like(r)
                    pil_img = Image.merge("RGB", (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))
                else:
                    # Reshape from (C, H, W) to (H, W, C)
                    transposed = np.transpose(normalized, (1, 2, 0))
                    pil_img = Image.fromarray(transposed)
                    
                pil_img.save(output_path, "PNG")
                
        except Exception as e:
            # Fallback preview (generate a blank image with error message or simple grey block)
            fallback_img = Image.new("RGB", (256, 256), color=(40, 40, 60))
            fallback_img.save(output_path, "PNG")
            
        return output_path

    def validate_pair(self, asset1_id: str, asset2_id: str) -> PairValidationResult:
        """Validate if two assets are compatible for comparison (Optical-SAR or bi-temporal T1-T2)."""
        asset1 = self.get_asset(asset1_id)
        asset2 = self.get_asset(asset2_id)
        
        if not asset1 or not asset2:
            return PairValidationResult(
                pair_type="unknown",
                compatible=False,
                spatial_overlap=0.0,
                crs_compatible=False,
                resolution_compatible=False,
                temporal_valid=False,
                coregistration_status="FAIL",
                errors=["One or both assets could not be found."],
                warnings=[]
            )
            
        errors: List[str] = []
        warnings: List[str] = []
        
        # 1. Determine Pair Type
        pair_type = "unknown"
        is_optical_1 = asset1.modality == "optical" or asset1.modality == "multispectral"
        is_optical_2 = asset2.modality == "optical" or asset2.modality == "multispectral"
        is_sar_1 = asset1.modality == "sar"
        is_sar_2 = asset2.modality == "sar"
        
        if (is_optical_1 and is_sar_2) or (is_sar_1 and is_optical_2):
            pair_type = "optical_sar"
        elif (is_optical_1 and is_optical_2) or (is_sar_1 and is_sar_2):
            pair_type = "t1_t2"
            
        # 2. CRS Compatibility
        crs_compatible = True
        both_unknown_crs = (asset1.crs == "UNKNOWN" and asset2.crs == "UNKNOWN")
        if asset1.crs == "UNKNOWN" or asset2.crs == "UNKNOWN":
            if both_unknown_crs:
                # Both images lack CRS — check if pixel grids match (co-registered benchmark pairs)
                same_dims = (asset1.width == asset2.width) and (asset1.height == asset2.height)
                same_transform = np.allclose(asset1.transform, asset2.transform, rtol=1e-5)
                if same_dims and same_transform:
                    crs_compatible = True  # Treat as compatible in pixel space
                    warnings.append("Both images lack CRS information. Pixel grid dimensions and transforms match — treating as co-registered pair.")
                else:
                    crs_compatible = False
                    errors.append("Both images lack CRS and have different dimensions or transforms.")
            else:
                crs_compatible = False
                errors.append("CRS information is missing from one or both images. Cannot match geospatial grids.")
        elif asset1.crs != asset2.crs:
            crs_compatible = False
            errors.append(f"CRS mismatch: {asset1.crs} vs {asset2.crs}.")
            
        # 3. Spatial Overlap Calculation
        spatial_overlap = 0.0
        if crs_compatible:
            # Bounds format: [left, bottom, right, top]
            b1 = asset1.bounds
            b2 = asset2.bounds
            
            # For non-georeferenced pairs with matching dimensions, compute pixel-space overlap
            # Normalize bounds so they are consistently [left, bottom, right, top] with bottom < top
            b1_norm = [b1[0], min(b1[1], b1[3]), b1[2], max(b1[1], b1[3])]
            b2_norm = [b2[0], min(b2[1], b2[3]), b2[2], max(b2[1], b2[3])]
            
            overlap_left = max(b1_norm[0], b2_norm[0])
            overlap_bottom = max(b1_norm[1], b2_norm[1])
            overlap_right = min(b1_norm[2], b2_norm[2])
            overlap_top = min(b1_norm[3], b2_norm[3])
            
            if overlap_left < overlap_right and overlap_bottom < overlap_top:
                overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
                area1 = (b1_norm[2] - b1_norm[0]) * (b1_norm[3] - b1_norm[1])
                area2 = (b2_norm[2] - b2_norm[0]) * (b2_norm[3] - b2_norm[1])
                min_area = min(area1, area2)
                
                if min_area > 0:
                    spatial_overlap = (overlap_area / min_area) * 100.0
            else:
                same_dims = (asset1.width == asset2.width) and (asset1.height == asset2.height)
                same_transform = np.allclose(asset1.transform, asset2.transform, rtol=1e-5)
                if both_unknown_crs and same_dims and same_transform:
                    # Same pixel dimensions and transforms = 100% overlap in pixel space
                    spatial_overlap = 100.0
                else:
                    spatial_overlap = 0.0
                    errors.append("No spatial overlap between the two images.")
        else:
            warnings.append("Spatial overlap could not be calculated due to missing/mismatched CRS.")
            
        # 4. Resolution Compatibility
        resolution_compatible = True
        # If resolutions are set/valid
        if asset1.resolution != [0.0, 0.0] and asset2.resolution != [0.0, 0.0]:
            resolution_compatible = np.allclose(asset1.resolution, asset2.resolution, rtol=0.15)
            if not resolution_compatible:
                warnings.append(f"Resolution mismatch: {asset1.resolution} vs {asset2.resolution}. Imagery may need resampling.")
        else:
            resolution_compatible = False
            warnings.append("Resolution metrics are missing or invalid.")
            
        # 5. Temporal Order Validation
        temporal_valid = True
        if asset1.timestamp and asset2.timestamp:
            if pair_type == "t1_t2":
                temporal_valid = asset1.timestamp <= asset2.timestamp
                if not temporal_valid:
                    errors.append(f"Temporal order invalid: T1 image acquisition time ({asset1.timestamp.isoformat()}) must be before or equal to T2 ({asset2.timestamp.isoformat()}).")
        else:
            # If dates are missing, fallback to true but warn
            warnings.append("Temporal acquisition dates are missing from image metadata. Order cannot be verified.")
            
        # 6. Co-registration Status
        # Do not claim exact co-registration based only on matching metadata
        coregistration_status = "NOT_VERIFIED"
        
        if len(errors) > 0:
            coregistration_status = "FAIL"
        else:
            # Metadata overlaps and matches
            if crs_compatible and spatial_overlap >= 99.0 and resolution_compatible:
                # Same bounds, same dimensions?
                same_dims = (asset1.width == asset2.width) and (asset1.height == asset2.height)
                # Same transform?
                same_transform = np.allclose(asset1.transform, asset2.transform, rtol=1e-5)
                
                if same_dims and same_transform:
                    coregistration_status = "NOT_VERIFIED"
                    warnings.append("Grid metadata matches perfectly. Actual pixel co-registration is not verified.")
                else:
                    coregistration_status = "WARNING"
                    warnings.append("Bounds match but grid alignments (dimensions or pixel transforms) differ slightly. Re-alignment required.")
            elif crs_compatible and spatial_overlap >= 50.0:
                coregistration_status = "WARNING"
                warnings.append(f"Images partially overlap ({spatial_overlap:.1f}%). Spatial clipping and alignment will be required.")
            else:
                coregistration_status = "FAIL"
                errors.append(f"Images have insufficient overlap ({spatial_overlap:.1f}%).")
                
        compatible = (len(errors) == 0) and (coregistration_status != "FAIL")
        
        return PairValidationResult(
            pair_type=pair_type,
            compatible=compatible,
            spatial_overlap=spatial_overlap,
            crs_compatible=crs_compatible,
            resolution_compatible=resolution_compatible,
            temporal_valid=temporal_valid,
            coregistration_status=coregistration_status,
            warnings=warnings,
            errors=errors,
        )

    def load_demo_pair(self, sample_name: Optional[str] = None) -> Tuple[RasterAsset, RasterAsset]:
        """Loads authentic LEVIR-CD sample images into the workspace as active assets."""
        import random
        from pathlib import Path

        available_samples = [
            "test_121_0768_0256",
            "test_2_0000_0000",
            "test_2_0000_0512",
            "test_55_0256_0000",
            "test_77_0512_0256",
            "test_7_0256_0512",
            "train_36_0512_0512",
            "train_386_0512_0768",
            "train_412_0512_0768",
            "val_27_0000_0256",
            "test_102_0512_0000",
        ]

        if not sample_name or sample_name == "random" or sample_name == "test_102_0512_0000":
            sample_name = random.choice(available_samples)
        
        # Project root is two levels up from this file (backend/app/services/ -> project root)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Search candidate paths for genuine demo samples
        t1_candidates = [
            project_root / f"sample_test_images/T1_old/{sample_name}.png",
            Path(f"sample_test_images/T1_old/{sample_name}.png"),
            Path(f"../sample_test_images/T1_old/{sample_name}.png"),
            Path(f"datasets/levir_cd/A/{sample_name}.png"),
            Path(f"../datasets/levir_cd/A/{sample_name}.png"),
        ]
        t2_candidates = [
            project_root / f"sample_test_images/T2_new/{sample_name}.png",
            Path(f"sample_test_images/T2_new/{sample_name}.png"),
            Path(f"../sample_test_images/T2_new/{sample_name}.png"),
            Path(f"datasets/levir_cd/B/{sample_name}.png"),
            Path(f"../datasets/levir_cd/B/{sample_name}.png"),
        ]
        
        t1_source = next((p for p in t1_candidates if p.exists()), None)
        t2_source = next((p for p in t2_candidates if p.exists()), None)
        
        if not t1_source or not t2_source:
            # Fallback to first available sample
            sample_name = "test_121_0768_0256"
            t1_source = project_root / f"sample_test_images/T1_old/{sample_name}.png"
            t2_source = project_root / f"sample_test_images/T2_new/{sample_name}.png"
            
        if not t1_source.exists() or not t2_source.exists():
            raise FileNotFoundError(f"Authentic LEVIR-CD sample '{sample_name}' not found on disk.")
            
        t1_id = f"demo_t1_{uuid.uuid4().hex[:8]}"
        t2_id = f"demo_t2_{uuid.uuid4().hex[:8]}"
        
        t1_dir = settings.upload_dir / t1_id
        t2_dir = settings.upload_dir / t2_id
        t1_dir.mkdir(parents=True, exist_ok=True)
        t2_dir.mkdir(parents=True, exist_ok=True)
        
        t1_dest = t1_dir / f"{sample_name}_t1.png"
        t2_dest = t2_dir / f"{sample_name}_t2.png"
        
        shutil.copyfile(t1_source, t1_dest)
        shutil.copyfile(t2_source, t2_dest)
        
        t1_asset = self.process_and_validate(t1_id, t1_dest, f"T1_Pre_{sample_name}.png")
        t2_asset = self.process_and_validate(t2_id, t2_dest, f"T2_Post_{sample_name}.png")
        
        # Generate thumbnails synchronously
        try:
            self.generate_preview(t1_id)
            self.generate_preview(t2_id)
        except Exception:
            pass
            
        return t1_asset, t2_asset


raster_ingestion_service = RasterIngestionService()

