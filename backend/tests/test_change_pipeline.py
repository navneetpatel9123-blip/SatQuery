"""
End-to-End Integration Tests for Phase 4 Change Understanding Pipeline.
"""
import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from typing import Dict, Any

from app.schemas import ChangeRegion, ChangeType, RasterAsset
from app.services.change_pipeline import change_pipeline
from app.config import settings

def _create_synthetic_geotiff(path: str, bands: list[np.ndarray]):
    """Create a temporary GeoTIFF with the given bands."""
    num_bands = len(bands)
    height, width = bands[0].shape
    transform = from_origin(10.0, 10.0, 1.0, 1.0)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=bands[0].dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        for i, (band, desc) in enumerate(zip(bands, ["red", "green", "blue", "nir", "swir"]), start=1):
            dst.write(band, i)
            dst.set_band_description(i, desc)

@pytest.fixture
def synthetic_t1_t2_assets(tmp_path):
    """Fixture to generate synthetic T1/T2 GeoTIFFs and return RasterAssets."""
    settings.upload_dir = tmp_path / "uploads"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    
    t1_dir = settings.upload_dir / "t1_asset"
    t1_dir.mkdir()
    t1_path = t1_dir / "t1.tif"
    
    t2_dir = settings.upload_dir / "t2_asset"
    t2_dir.mkdir()
    t2_path = t2_dir / "t2.tif"
    
    # T1: Vegetation
    t1_bands = [
        np.full((8, 8), 30.0),   # Red
        np.full((8, 8), 60.0),   # Green
        np.full((8, 8), 40.0),   # Blue
        np.full((8, 8), 200.0),  # NIR
        np.full((8, 8), 50.0),   # SWIR
    ]
    
    # T2: Built-up
    t2_bands = [
        np.full((8, 8), 100.0),  # Red
        np.full((8, 8), 90.0),   # Green
        np.full((8, 8), 85.0),   # Blue
        np.full((8, 8), 120.0),  # NIR
        np.full((8, 8), 150.0),  # SWIR
    ]
    
    _create_synthetic_geotiff(str(t1_path), t1_bands)
    _create_synthetic_geotiff(str(t2_path), t2_bands)
    
    t1_asset = RasterAsset(
        id="t1_asset",
        filename="t1.tif",
        path=str(t1_path),
        modality="multispectral",
        modality_certain=True,
        width=8,
        height=8,
        bands=5,
        dtype="float64",
        crs="EPSG:4326",
        bounds=[10.0, 2.0, 18.0, 10.0],
        resolution=[1.0, 1.0],
        transform=[1.0, 0.0, 10.0, 0.0, -1.0, 10.0],
        validation_status="PASS"
    )
    
    t2_asset = RasterAsset(
        id="t2_asset",
        filename="t2.tif",
        path=str(t2_path),
        modality="multispectral",
        modality_certain=True,
        width=8,
        height=8,
        bands=5,
        dtype="float64",
        crs="EPSG:4326",
        bounds=[10.0, 2.0, 18.0, 10.0],
        resolution=[1.0, 1.0],
        transform=[1.0, 0.0, 10.0, 0.0, -1.0, 10.0],
        validation_status="PASS"
    )
    
    return t1_asset, t2_asset

def test_change_pipeline_end_to_end(synthetic_t1_t2_assets):
    t1_asset, t2_asset = synthetic_t1_t2_assets
    
    change_regions = [
        ChangeRegion(
            region_id="P3-001",
            change_type=ChangeType.OTHER,
            area_sq_m=64.0,
            centroid=[14.0, 6.0],
            bbox=[0.0, 0.0, 1.0, 1.0],
            confidence=0.85,
            description="Phase 3 detected a change."
        )
    ]
    
    result = change_pipeline.run_pipeline(
        t1_asset=t1_asset,
        t2_asset=t2_asset,
        change_regions=change_regions
    )
    
    assert result.phase == "evidence_extraction"
    assert result.models_used[0]["model_id"] == "spectral_land_cover_classifier"
    assert result.models_used[1]["model_id"] == "rule_based_change_understander"
    
    d = result.to_dict()
    assert d["analysis_id"] == result.analysis_id
    assert len(d["regions"]) == 1
    
    region = d["regions"][0]
    assert region["region_id"] == "P3-001"
    assert region["t1_land_cover"] == "VEGETATION"
    assert region["t2_land_cover"] == "BUILT_UP"
    assert region["change_type"] == "BUILT_UP_EXPANSION"
    assert region["change_confidence"]["score"] > 0
    assert len(region["evidence"]) > 0
    assert len(region["description"]) > 0
    
    # Check execution trace
    events = [te["event"] for te in d["execution_trace"]]
    assert "PHASE_3_COMPLETED" in events
    assert "CHANGE_REGIONS_AVAILABLE" in events
    assert "PHASE_4_STARTED" in events
    assert "LAND_COVER_CLASSIFICATION" in events
    assert "CHANGE_UNDERSTANDING" in events
    assert "CONFIDENCE_CALCULATION" in events
    assert "PHASE_4_COMPLETED" in events
