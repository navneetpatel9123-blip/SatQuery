import os
import uuid
import rasterio
from rasterio.plot import reshape_as_image
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple

from app.config import settings
from app.schemas import GeoTIFFMetadata, ImagePairValidation


class GeoTIFFService:
    def __init__(self):
        pass

    def extract_metadata(self, file_path: Path) -> GeoTIFFMetadata:
        """Extract metadata from a GeoTIFF file."""
        with rasterio.open(file_path) as src:
            bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            # Handle resolution depending on version/properties available, normally src.res
            res = list(src.res) if hasattr(src, "res") else [0.0, 0.0]
            
            # Simple datetime extraction attempt from tags
            tags = src.tags()
            capture_date = None
            if "TIFFTAG_DATETIME" in tags:
                try:
                    capture_date = datetime.strptime(tags["TIFFTAG_DATETIME"], "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass

            return GeoTIFFMetadata(
                filename=file_path.name,
                crs=src.crs.to_string() if src.crs else "UNKNOWN",
                bounds=bounds,
                resolution=res,
                width=src.width,
                height=src.height,
                band_count=src.count,
                dtype=str(src.dtypes[0]) if src.dtypes else "unknown",
                capture_date=capture_date,
                file_size_mb=file_path.stat().st_size / (1024 * 1024)
            )

    def validate_pair(self, t1_path: Path, t2_path: Path) -> ImagePairValidation:
        """Validate if two GeoTIFFs can be compared."""
        meta1 = self.extract_metadata(t1_path)
        meta2 = self.extract_metadata(t2_path)

        errors = []
        warnings = []
        
        # CRS Match
        crs_match = meta1.crs == meta2.crs
        if not crs_match:
            errors.append(f"CRS mismatch: {meta1.crs} vs {meta2.crs}")

        # Bounds Overlap Check
        def get_overlap_area(b1, b2):
            dx = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            dy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            return dx * dy

        area1 = (meta1.bounds[2] - meta1.bounds[0]) * (meta1.bounds[3] - meta1.bounds[1])
        area2 = (meta2.bounds[2] - meta2.bounds[0]) * (meta2.bounds[3] - meta2.bounds[1])
        
        overlap_area = get_overlap_area(meta1.bounds, meta2.bounds)
        min_area = min(area1, area2) if area1 > 0 and area2 > 0 else 0
        
        bounds_overlap_pct = (overlap_area / min_area * 100) if min_area > 0 else 0
        
        same_aoi = bounds_overlap_pct >= 95.0
        if not same_aoi:
            errors.append(f"Insufficient overlap: {bounds_overlap_pct:.1f}%")

        # Temporal ordering
        temporal_order_valid = True
        if meta1.capture_date and meta2.capture_date:
            temporal_order_valid = meta1.capture_date < meta2.capture_date
            if not temporal_order_valid:
                errors.append("T1 capture date is not before T2")
        else:
            warnings.append("Missing capture date metadata. Relying on user ordering.")

        # Resolution compatibility
        resolution_compatible = np.allclose(meta1.resolution, meta2.resolution, rtol=0.1)
        if not resolution_compatible:
            warnings.append(f"Resolution mismatch: {meta1.resolution} vs {meta2.resolution}")

        is_valid = crs_match and same_aoi and temporal_order_valid

        return ImagePairValidation(
            is_valid=is_valid,
            crs_match=crs_match,
            bounds_overlap_pct=bounds_overlap_pct,
            same_aoi=same_aoi,
            temporal_order_valid=temporal_order_valid,
            resolution_compatible=resolution_compatible,
            errors=errors,
            warnings=warnings
        )

    def generate_thumbnail(self, tiff_path: Path, output_path: Path, size: Tuple[int, int] = (512, 512)):
        """Generate a downsampled PNG thumbnail for a GeoTIFF."""
        with rasterio.open(tiff_path) as src:
            # Read first 3 bands (RGB) or single band if grayscale
            num_bands = min(src.count, 3)
            
            # Use out_shape to downsample during read
            out_shape = (num_bands, size[1], size[0])
            img_array = src.read(
                indexes=list(range(1, num_bands + 1)),
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.bilinear
            )

            # Normalize to 0-255 uint8
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.float32)
                # Simple percentiles to ignore outliers
                p2, p98 = np.percentile(img_array, (2, 98))
                img_array = np.clip((img_array - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            
            # Reshape for PIL (H, W, C)
            if num_bands == 1:
                img_array = img_array[0] # H, W
            else:
                img_array = reshape_as_image(img_array) # H, W, C
                
            img = Image.fromarray(img_array)
            img.save(output_path, "PNG")


geotiff_service = GeoTIFFService()
