import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.schemas import RasterAsset
from app.services.pipeline_orchestrator import (
    TaskClassifier,
    QueryTask,
    InputValidator,
    WorkflowPlanner,
    OrchestratorRequest,
    orchestrator
)
from app.config import settings

def test_query_classification():
    assert TaskClassifier.classify("What changed between these two dates?")[0] == QueryTask.BI_TEMPORAL_CHANGE
    assert TaskClassifier.classify("What type of change occurred?")[0] == QueryTask.BI_TEMPORAL_CHANGE_UNDERSTANDING
    assert TaskClassifier.classify("Give evidence for the detected change.")[0] == QueryTask.BI_TEMPORAL_EVIDENCE
    assert TaskClassifier.classify("Describe this satellite image.")[0] == QueryTask.SINGLE_IMAGE_CAPTION
    assert TaskClassifier.classify("What is visible in this image?")[0] == QueryTask.SINGLE_IMAGE_VQA
    assert TaskClassifier.classify("Highlight the water body.")[0] == QueryTask.SINGLE_IMAGE_GROUNDING
    assert TaskClassifier.classify("Use optical and SAR together.")[0] == QueryTask.OPTICAL_SAR_ANALYSIS
    assert TaskClassifier.classify("hello world")[0] == QueryTask.UNKNOWN_TASK
    assert TaskClassifier.classify("What is the meaning of life?")[0] == QueryTask.UNKNOWN_TASK

def test_ambiguous_query_classification():
    # Simple ambiguity
    task, msg = TaskClassifier.classify("What is the change and why? It's weird")
    assert task == QueryTask.BI_TEMPORAL_EVIDENCE

def test_input_validation():
    # Missing images
    req = OrchestratorRequest(query="detect change")
    valid, msg, assets = InputValidator.validate(QueryTask.BI_TEMPORAL_CHANGE, req)
    assert not valid
    assert "need T1 and T2" in msg
    
    # Invalid optical sar
    req = OrchestratorRequest(query="Use optical and SAR together.")
    valid, msg, assets = InputValidator.validate(QueryTask.OPTICAL_SAR_ANALYSIS, req)
    assert not valid
    assert "requires exactly one OPTICAL and one SAR image" in msg

def test_workflow_planning():
    assert WorkflowPlanner.plan(QueryTask.BI_TEMPORAL_CHANGE) == ["CHANGE_DETECTION"]
    assert WorkflowPlanner.plan(QueryTask.BI_TEMPORAL_CHANGE_UNDERSTANDING) == ["CHANGE_DETECTION", "CHANGE_UNDERSTANDING"]
    assert WorkflowPlanner.plan(QueryTask.BI_TEMPORAL_EVIDENCE) == ["CHANGE_DETECTION", "CHANGE_UNDERSTANDING", "EVIDENCE_EXTRACTION"]
    assert WorkflowPlanner.plan(QueryTask.SINGLE_IMAGE_VQA) == ["VQA"]
    assert WorkflowPlanner.plan(QueryTask.SINGLE_IMAGE_CAPTION) == ["CAPTIONING"]

def _create_synthetic_geotiff(path: str, bands: list[np.ndarray]):
    num_bands = len(bands)
    height, width = bands[0].shape
    transform = from_origin(10.0, 10.0, 1.0, 1.0)
    with rasterio.open(
        path, 'w', driver='GTiff', height=height, width=width,
        count=num_bands, dtype=bands[0].dtype, crs='EPSG:4326', transform=transform
    ) as dst:
        for i, (band, desc) in enumerate(zip(bands, ["red", "green", "blue", "nir", "swir"]), start=1):
            dst.write(band, i)
            dst.set_band_description(i, desc)

@pytest.fixture
def orchestrator_assets(tmp_path):
    settings.upload_dir = tmp_path / "uploads"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    t1_path = settings.upload_dir / "t1_orch.tif"
    t2_path = settings.upload_dir / "t2_orch.tif"
    
    # Vegetation
    t1_bands = [
        np.full((8, 8), 30.0), np.full((8, 8), 60.0), np.full((8, 8), 40.0),
        np.full((8, 8), 200.0), np.full((8, 8), 50.0),
    ]
    # Built-up
    t2_bands = [
        np.full((8, 8), 100.0), np.full((8, 8), 90.0), np.full((8, 8), 85.0),
        np.full((8, 8), 120.0), np.full((8, 8), 150.0),
    ]
    
    _create_synthetic_geotiff(str(t1_path), t1_bands)
    _create_synthetic_geotiff(str(t2_path), t2_bands)
    
    from app.services.raster_ingestion_service import raster_ingestion_service
    a1 = raster_ingestion_service.process_and_validate("t1_orch_mock_id", t1_path, "t1_orch.tif")
    a2 = raster_ingestion_service.process_and_validate("t2_orch_mock_id", t2_path, "t2_orch.tif")
    
    return a1, a2

def test_unavailable_capability_handling(orchestrator_assets):
    # Test with SINGLE_IMAGE_GROUNDING which is still unimplemented
    req = OrchestratorRequest(query="ground this area", t1_image_id=orchestrator_assets[0].id)
    resp = orchestrator.execute(req)
    
    assert resp.status == "UNAVAILABLE"
    assert "not currently available" in resp.warnings[0]
    assert "ORCHESTRATION_FAILED" in [t["event"] for t in resp.execution_trace]
    
    # Security: Unregistered tools are not planned
    events = [t["event"] for t in resp.execution_trace]
    assert "ORCHESTRATION_STARTED" in events
    assert "QUERY_CLASSIFIED" in events
    assert "ORCHESTRATION_FAILED" in events

def test_missing_t1_t2_handling():
    req = OrchestratorRequest(query="what changed between these two dates?", t1_image_id="fake_id")
    resp = orchestrator.execute(req)
    assert resp.status == "NEEDS_CLARIFICATION"
    assert "need T1 and T2" in resp.warnings[0]

def test_end_to_end_orchestrator(orchestrator_assets):
    a1, a2 = orchestrator_assets
    
    # End to End: Phase 3 -> 4 -> 5 via Natural Language
    req = OrchestratorRequest(
        query="Why do you think this is built-up expansion?",
        t1_image_id=a1.id,
        t2_image_id=a2.id
    )
    
    resp = orchestrator.execute(req)
    print("WARNINGS:", resp.warnings)
    print("TRACE:", [t["event"] for t in resp.execution_trace])
    assert resp.status == "COMPLETED"
    assert resp.task == "BI_TEMPORAL_EVIDENCE"
    assert resp.workflow == ["CHANGE_DETECTION", "CHANGE_UNDERSTANDING", "EVIDENCE_EXTRACTION"]
    
    # Verify execution trace
    events = [t["event"] for t in resp.execution_trace]
    assert "ORCHESTRATION_STARTED" in events
    assert "QUERY_CLASSIFIED" in events
    assert "INPUT_VALIDATED" in events
    assert "WORKFLOW_PLANNED" in events
    assert "STEP_STARTED" in events
    assert "STEP_COMPLETED" in events
    assert "PHASE_4_STARTED" in events # Internal change_pipeline trace
    assert "ORCHESTRATION_COMPLETED" in events
    
    # Verify results
    assert len(resp.results["regions"]) > 0
    region = resp.results["regions"][0]
    
    assert "evidence" in region
    assert region["change_type"] is not None
    assert region["evidence"]["evidence_summary"]["overall_strength"] is not None
