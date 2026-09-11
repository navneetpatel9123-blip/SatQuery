import pytest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from app.services.pipeline_orchestrator import PipelineOrchestrator, OrchestratorRequest
from app.services.raster_ingestion_service import RasterAsset, raster_ingestion_service

def _create_synthetic_geotiff(path: str, bands: int = 4):
    data = np.zeros((bands, 10, 10), dtype=np.float32)
    transform = from_origin(0.0, 10.0, 1.0, 1.0)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=10,
        width=10,
        count=bands,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data)

@pytest.fixture
def orchestrator():
    return PipelineOrchestrator()

@pytest.fixture
def orchestrator_vqa_asset(tmp_path):
    t1_path = tmp_path / "t1_vqa.tif"
    _create_synthetic_geotiff(str(t1_path), 4)
    a1 = raster_ingestion_service.process_and_validate("t1_vqa_mock_id", t1_path, "t1_vqa.tif")
    return a1

def test_end_to_end_vqa_orchestration(orchestrator, orchestrator_vqa_asset):
    req = OrchestratorRequest(query="Is there water visible in this image?", t1_image_id=orchestrator_vqa_asset.id)
    resp = orchestrator.execute(req)
    
    print("VQA TRACE:", [t["event"] for t in resp.execution_trace])
    print("VQA WARNINGS:", resp.warnings)
    
    assert resp.status == "COMPLETED"
    assert resp.task == "SINGLE_IMAGE_VQA"
    assert resp.workflow == ["VQA"]
    
    assert "answer" in resp.results
    assert resp.results["confidence"] > 0.0
    assert resp.evidence is not None
    
    trace_events = [t["event"] for t in resp.execution_trace]
    assert "VQA_STARTED" in trace_events
    assert "VQA_COMPLETED" in trace_events

def test_end_to_end_caption_orchestration(orchestrator, orchestrator_vqa_asset):
    req = OrchestratorRequest(query="Describe this image.", t1_image_id=orchestrator_vqa_asset.id)
    resp = orchestrator.execute(req)
    
    print("CAPTION TRACE:", [t["event"] for t in resp.execution_trace])
    print("CAPTION WARNINGS:", resp.warnings)
    
    assert resp.status == "COMPLETED"
    assert resp.task == "SINGLE_IMAGE_CAPTION"
    assert resp.workflow == ["CAPTIONING"]
    
    assert "caption" in resp.results
    assert resp.results["confidence"] >= 0.0
    
    trace_events = [t["event"] for t in resp.execution_trace]
    assert "CAPTION_STARTED" in trace_events
    assert "CAPTION_COMPLETED" in trace_events
