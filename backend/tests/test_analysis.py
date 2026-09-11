import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json

from app.main import app
from app.services.model_registry import model_registry
from app.services.agent_service import agent_service
from app.services.raster_ingestion_service import raster_ingestion_service
from app.schemas import RasterAsset, EvidenceResult
from tests.test_ingestion import create_synth_tiff

client = TestClient(app)


# 1. Test scenario: Query classification routing
def test_query_classification():
    # Captioning keywords
    assert agent_service.classify_task("Describe the scene overview of this image") == "CAPTIONING"
    assert agent_service.classify_task("Summarize the land cover visible") == "CAPTIONING"
    
    # Grounding keywords
    assert agent_service.classify_task("Highlight the water body on the map") == "GROUNDING"
    assert agent_service.classify_task("Locate the buildings using bounding boxes") == "GROUNDING"
    assert agent_service.classify_task("Where is the runway segment?") == "GROUNDING"
    
    # VQA default questions
    assert agent_service.classify_task("Is there a water body?") == "SINGLE_VQA"
    assert agent_service.classify_task("Are agricultural fields present?") == "SINGLE_VQA"


# 2. Test scenario: Model registry
def test_model_registry():
    models = model_registry.list_models()
    model_ids = [m.model_id for m in models]
    
    assert "rs_vqa_adapted" in model_ids
    assert "demo_fallback" in model_ids
    assert "base_vlm" in model_ids
    
    # Check updated metadata fields
    adapted_info = next(m for m in models if m.model_id == "rs_vqa_adapted")
    assert adapted_info.model_name == "SatQuery VQA Adapted ResNet50"
    assert adapted_info.framework == "PyTorch/TorchScript"
    assert adapted_info.remote_sensing_adapted is True
    assert adapted_info.training_dataset == "BigEarthNet-19"
    assert adapted_info.input_format == "Multispectral GeoTIFF (RGB+NIR)"
    assert adapted_info.checkpoint == "resnet50_bigearthnet_sih26167.pth"
    
    # Test lazy loading retrieves loaded model
    vqa_model = model_registry.get_model("rs_vqa_adapted")
    assert vqa_model is not None
    assert vqa_model.info.status == "LOADED"
    
    # Assert lifecycle methods
    assert vqa_model.health_check() is True
    vqa_model.unload()
    assert vqa_model._is_loaded is False
    assert vqa_model.info.status == "STANDBY"


# 3. Test scenario: Adapters & fallback behavior
def test_demo_fallback_adapters():
    demo_model = model_registry.get_model("demo_fallback")
    assert demo_model is not None
    assert demo_model.health_check() is True
    
    # Mock asset
    mock_asset = RasterAsset(
        id="test-asset-vqa",
        filename="test.tif",
        path="data/uploads/test-asset-vqa/test.tif",
        modality="optical",
        modality_certain=True,
        width=100,
        height=100,
        bands=3,
        dtype="uint8",
        crs="EPSG:4326",
        bounds=[-122.5, 37.7, -122.4, 37.8],
        resolution=[0.0001, 0.0001],
        transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        validation_status="PASS"
    )
    
    # Assert validation
    assert demo_model.validate_input(mock_asset) is True
    
    # VQA
    ans, conf, evidence, warnings = demo_model.predict(mock_asset, "Is there a water body?")
    assert "water body" in ans
    assert conf > 0.90
    assert len(evidence) > 0
    assert evidence[0].type == "spatial"
    
    # Captioning
    caption_dict, caption_conf, caption_ev, caption_warns = demo_model.generate_caption(mock_asset)
    assert "SCENE OVERVIEW" in caption_dict
    assert "LAND COVER" in caption_dict
    assert "MAJOR OBJECTS" in caption_dict
    
    # Grounding
    ground_ev, ground_conf, ground_warns = demo_model.ground_text(mock_asset, "Highlight the road")
    assert len(ground_ev) > 0
    assert ground_ev[0].bbox is not None
    assert ground_conf > 0.70


# 3.5 Test scenario: Query normalization & conversational follow-ups
def test_query_normalization_and_conversational_history(tmp_path):
    # Setup test file
    file_path = tmp_path / "conv_test.tif"
    create_synth_tiff(file_path, bands=3)
    raster_ingestion_service.process_and_validate("conv-asset-id", file_path, "conv_test.tif")
    
    # Query Normalization
    assert agent_service.normalize_query("Can you see water?") == "is there water"
    assert agent_service.normalize_query("Are there any water bodies?") == "is there water"
    assert agent_service.normalize_query("Identify agricultural plots") == "identify agricultural land"
    
    # Turn 1: VQA question about water
    session = "test-session-1"
    resp1 = agent_service.run_analysis(
        image_id="conv-asset-id",
        query="Is there a water body?",
        session_id=session
    )
    assert resp1.task == "SINGLE_VQA"
    
    # Turn 2: Follow-up question referencing "it"
    resp2 = agent_service.run_analysis(
        image_id="conv-asset-id",
        query="Where is it?",
        session_id=session
    )
    # Check that it resolved "it" to water and routed to GROUNDING
    assert resp2.task == "GROUNDING"
    assert "water body" in resp2.answer.lower() or "localized" in resp2.answer.lower()
    assert len(resp2.evidence) > 0
    assert resp2.evidence[0].bbox is not None


# 4. Test scenario: Agent routing & trace generation
def test_agent_routing_and_trace(tmp_path):
    # Set up synthetic asset in system
    file_path = tmp_path / "agent_test.tif"
    create_synth_tiff(file_path, bands=3)
    asset = raster_ingestion_service.process_and_validate("agent-asset-id", file_path, "agent_test.tif")
    
    # Query for scene description -> routes to Captioning on demo_fallback
    response = agent_service.run_analysis("agent-asset-id", "Describe this image")
    
    assert response.task == "CAPTIONING"
    assert "SCENE OVERVIEW" in response.answer
    assert response.confidence["level"] == "HIGH"
    assert len(response.execution_trace) == 4 # 4 pipeline nodes
    
    # Verify trace flow
    trace_steps = [node.step_id for node in response.execution_trace]
    assert "task_classification" in trace_steps
    assert "modality_check" in trace_steps
    assert "model_inference" in trace_steps
    assert "evidence_synthesis" in trace_steps


# 5. Test scenario: Modality constraint check
def test_unsupported_modality_constraint(tmp_path):
    # Create a SAR image asset
    sar_path = tmp_path / "test_sar.tif"
    create_synth_tiff(sar_path, bands=1)
    raster_ingestion_service.process_and_validate("sar-asset-id", sar_path, "test_sar.tif")
    
    # Requesting rs_vqa_adapted on SAR should fail/fallback gracefully
    # RSAdaptedVQAModel itself raises ValueError for SAR modality
    vqa_model = model_registry.get_model("rs_vqa_adapted")
    
    sar_asset = raster_ingestion_service.get_asset("sar-asset-id")
    with pytest.raises(ValueError) as val_err:
        vqa_model.answer_question(sar_asset, "Is there water?")
    assert "does not currently support SAR" in str(val_err.value)
    
    # Running via the Agent Service handles the exception and falls back to a warning/errors block gracefully
    response = agent_service.run_analysis("sar-asset-id", "Is there water?", preferred_model_id="rs_vqa_adapted")
    
    # Should engage fallback to demo fallback model
    assert response.model.model_id == "demo_fallback"
    assert any("does not currently support" in w or "failed" in w or "Fallback" in w for w in response.warnings)


# 6. Test scenario: API Endpoint integration testing
def test_analysis_endpoints(tmp_path):
    # Create valid synthetic image in ingestion uploads dir
    file_path = tmp_path / "api_test.tif"
    create_synth_tiff(file_path, bands=3)
    asset = raster_ingestion_service.process_and_validate("api-asset-id", file_path, "api_test.tif")
    
    # A. GET models registry
    res_models = client.get("/api/v1/models")
    assert res_models.status_code == 200
    model_ids = [m["model_id"] for m in res_models.json()]
    assert "rs_vqa_adapted" in model_ids
    
    # B. POST Agent Query (VQA)
    payload = {
        "image_id": "api-asset-id",
        "query": "Is there agricultural land?",
        "model_id": "rs_vqa_adapted"
    }
    res_query = client.post("/api/v1/analysis/query", json=payload)
    assert res_query.status_code == 200
    data = res_query.json()
    assert data["task"] == "SINGLE_VQA"
    assert "answer" in data
    assert "evidence" in data
    assert "confidence" in data
    assert "execution_trace" in data
    
    analysis_id = data["analysis_id"]
    
    # C. GET Analysis historical details
    res_hist = client.get(f"/api/v1/analysis/{analysis_id}")
    assert res_hist.status_code == 200
    assert res_hist.json()["answer"] == data["answer"]
    
    # D. POST Direct VQA
    vqa_payload = {
        "image_id": "api-asset-id",
        "query": "Is there a water body?"
    }
    res_vqa = client.post("/api/v1/analysis/vqa", json=vqa_payload)
    assert res_vqa.status_code == 200
    assert res_vqa.json()["task"] == "SINGLE_VQA"
    
    # E. POST Direct Captioning
    cap_payload = {
        "image_id": "api-asset-id"
    }
    res_cap = client.post("/api/v1/analysis/caption", json=cap_payload)
    assert res_cap.status_code == 200
    assert res_cap.json()["task"] == "CAPTIONING"
    assert "SCENE OVERVIEW" in res_cap.json()["answer"]
    
    # F. POST Direct Grounding
    gr_payload = {
        "image_id": "api-asset-id",
        "query": "Highlight the road"
    }
    res_gr = client.post("/api/v1/analysis/ground", json=gr_payload)
    assert res_gr.status_code == 200
    assert res_gr.json()["task"] == "GROUNDING"
