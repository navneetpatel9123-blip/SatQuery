"""
End-to-End Integration Tests for Phase 5 Evidence Pipeline Integration.
"""
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

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


def test_evidence_pipeline_end_to_end(synthetic_t1_t2_assets):
    t1_asset, t2_asset = synthetic_t1_t2_assets
    
    # Phase 3 mock input
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
    
    # Run the pipeline which now executes Phase 5
    result = change_pipeline.run_pipeline(
        t1_asset=t1_asset,
        t2_asset=t2_asset,
        change_regions=change_regions
    )
    
    assert result.phase == "evidence_extraction"
    d = result.to_dict()
    
    assert "analysis_id" in d
    assert len(d["regions"]) == 1
    
    region = d["regions"][0]
    
    # Verify Phase 4 info is present
    assert region["region_id"] == "P3-001"
    assert region["t1_land_cover"] == "VEGETATION"
    assert region["t2_land_cover"] == "BUILT_UP"
    assert region["change_type"] == "BUILT_UP_EXPANSION"
    
    # Verify Phase 5 evidence is attached
    assert "evidence" in region
    ev = region["evidence"]
    
    # Check structure
    assert ev["evidence_summary"]["overall_strength"] in ["HIGH", "MEDIUM", "LOW", "INSUFFICIENT"]
    assert "score" in ev["evidence_summary"]
    
    assert "spatial_evidence" in ev
    assert "spectral_evidence" in ev
    assert "statistical_evidence" in ev
    assert "land_cover_evidence" in ev
    
    # Check spectral evidence actually computed real values
    ndbi_ev = next((s for s in ev["spectral_evidence"] if s["feature"] == "NDBI"), None)
    print("Spectral Evidence:", ev["spectral_evidence"])
    print("Pipeline Warnings:", d["warnings"])
    print("Models Used:", d["models"])
    
    print("DEBUG spectral_features_t1:", result.result.regions[0].spectral_features_t1)
    
    with rasterio.open(t1_asset.path) as src:
        print("T1 Descriptions:", src.descriptions)
        
    assert ndbi_ev is not None
    assert ndbi_ev["available"] is True
    assert ndbi_ev["delta"] > 0.05
    
    # Check supporting evidence
    assert len(ev["evidence_summary"]["supporting_evidence"]) > 0
    assert "explanation" in ev
    assert "built-up spectral indicator (NDBI)" in ev["explanation"]
    
    # Check execution trace
    events = [te["event"] for te in d["execution_trace"]]
    assert "PHASE_4_COMPLETED" in events
    assert "PHASE_5_STARTED" in events
    assert "EVIDENCE_EXTRACTION_STARTED" in events
    assert "EVIDENCE_RESULTS_ATTACHED" in events
    assert "PHASE_5_COMPLETED" in events
