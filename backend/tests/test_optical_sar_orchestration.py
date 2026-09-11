import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.schemas import RasterAsset
from app.services.pipeline_orchestrator import PipelineOrchestrator, OrchestratorRequest

def _create_raster(path, data, crs="EPSG:4326", transform=None, nodata=None, descriptions=None):
    if transform is None:
        transform = from_origin(0, 0, 10, 10)
    count, height, width = data.shape
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = tuple(descriptions)

@pytest.fixture
def orchestrator_assets(tmp_path, monkeypatch):
    # Create optical and SAR
    opt_path = tmp_path / "opt.tif"
    sar_path = tmp_path / "sar.tif"
    
    # Water (High NDWI: Green - NIR / Green + NIR)
    # Green = 150, NIR = 50 -> NDWI = 100/200 = 0.5
    opt_data = np.array([
        np.full((8, 8), 50.0),   # NIR
        np.full((8, 8), 150.0),  # Green
    ], dtype=np.float32)
    _create_raster(opt_path, opt_data, descriptions=["NIR", "Green"])
    
    # Low backscatter for water
    sar_data = np.array([np.full((8, 8), 0.02)], dtype=np.float32)
    _create_raster(sar_path, sar_data, descriptions=["VV"])
    
    from app.services.raster_ingestion_service import raster_ingestion_service
    # Mock asset generation
    opt = RasterAsset(
        id="opt_orch", filename="opt.tif", path=str(opt_path), modality="optical",
        modality_certain=True, width=8, height=8, bands=2, dtype="float32",
        crs="EPSG:4326", bounds=[0, -80, 80, 0], resolution=[10.0, 10.0],
        transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    sar = RasterAsset(
        id="sar_orch", filename="sar.tif", path=str(sar_path), modality="sar",
        modality_certain=True, width=8, height=8, bands=1, dtype="float32",
        crs="EPSG:4326", bounds=[0, -80, 80, 0], resolution=[10.0, 10.0],
        transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    
    def mock_get_asset(asset_id):
        if asset_id == "opt_orch": return opt
        if asset_id == "sar_orch": return sar
        return None
        
    monkeypatch.setattr(raster_ingestion_service, "get_asset", mock_get_asset)
    return opt, sar

@pytest.fixture
def orchestrator():
    return PipelineOrchestrator()

def test_end_to_end_optical_sar_orchestration(orchestrator, orchestrator_assets):
    opt, sar = orchestrator_assets
    req = OrchestratorRequest(
        query="Use optical and SAR together to identify water covered regions.",
        t1_image_id=opt.id,
        t2_image_id=sar.id
    )
    
    resp = orchestrator.execute(req)
    
    # Check trace
    events = [t["event"] for t in resp.execution_trace]
    assert "ORCHESTRATION_STARTED" in events
    assert "QUERY_CLASSIFIED" in events
    assert "OPTICAL_SAR_STARTED" in events
    assert "CROSS_MODAL_FUSION_STARTED" in events
    assert "OPTICAL_SAR_COMPLETED" in events
    
    assert resp.status == "COMPLETED"
    assert resp.task == "OPTICAL_SAR_ANALYSIS"
    assert "cross_modal_analysis" in resp.evidence
    
    regions = resp.results.get("regions", [])
    assert len(regions) == 1
    reg = regions[0]
    
    assert reg["classification"] == "WATER"
    assert reg["agreement"]["label"] == "STRONG_AGREEMENT"
    assert reg["sar_evidence"]["mean_backscatter"] == pytest.approx(0.02, rel=1e-5)
