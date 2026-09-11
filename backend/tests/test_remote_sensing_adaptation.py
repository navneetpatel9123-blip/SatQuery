"""
SatQuery AI — Remote Sensing Adaptation Tests
Unit tests for BigEarthNet dataset loader, preprocessing, adapted VLM, and fallback orchestration.
"""
import pytest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.schemas import RasterAsset
from app.services.pipeline_orchestrator import PipelineOrchestrator, OrchestratorRequest
from app.models.base_vlm import BaseRemoteSensingVisionModel
from app.models.adapted_remote_sensing_vlm import AdaptedRemoteSensingModel
from app.models.remote_sensing_vqa import RuleBasedRemoteSensingVQA
from app.models.remote_sensing_captioner import RuleBasedRemoteSensingCaptioner
from app.training.preprocessing import normalize_raster_bands, extract_rgb_nir
from app.training.dataset import BigEarthNetDataset, RSVQADatasetAdapter, VRSBenchDatasetAdapter, BIGEARTHNET_19_CLASSES
from app.services.model_registry import model_registry
from app.services.pipeline_orchestrator import PipelineOrchestrator
from app.services.raster_ingestion_service import raster_ingestion_service


def _create_synthetic_geotiff(path: str, bands: int = 4):
    data = np.full((bands, 10, 10), 100.0, dtype=np.float32)
    transform = from_origin(0.0, 10.0, 1.0, 1.0)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=10,
        width=10,
        count=bands,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def sample_asset(tmp_path):
    t_path = tmp_path / "sample_raster.tif"
    _create_synthetic_geotiff(str(t_path), 4)
    asset = raster_ingestion_service.process_and_validate("sample_asset_id", t_path, "sample_raster.tif")
    return asset


def test_base_interface_compliance():
    """Verify both rule-based and adapted models inherit/implement base properties."""
    vqa = RuleBasedRemoteSensingVQA()
    cap = RuleBasedRemoteSensingCaptioner()
    adapted = AdaptedRemoteSensingModel()

    assert issubclass(AdaptedRemoteSensingModel, BaseRemoteSensingVisionModel)
    assert vqa.is_trained is False
    assert vqa.adaptation_status == "RULE_BASED_BASELINE"
    assert cap.is_trained is False
    assert cap.adaptation_status == "RULE_BASED_BASELINE"
    assert adapted.is_trained is False
    assert adapted.adaptation_status == "TRAINING_READY"


def test_dataset_adapter_missing_directory_fails_honestly(tmp_path):
    """Verify BigEarthNet dataset loader fails with clear honest message when dataset is absent."""
    non_existent = tmp_path / "missing_bigearthnet"
    with pytest.raises(FileNotFoundError) as exc_info:
        BigEarthNetDataset(root_dir=str(non_existent))
    assert "REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED" in str(exc_info.value)


def test_benchmark_adapters_handle_missing_data():
    """Verify RSVQA and VRSBench report 'NOT RUN — DATASET NOT AVAILABLE' without crashing."""
    rsvqa = RSVQADatasetAdapter("/non/existent/rsvqa/path")
    vrsbench = VRSBenchDatasetAdapter("/non/existent/vrsbench/path")

    assert rsvqa.available is False
    assert rsvqa.status_message == "NOT RUN — DATASET NOT AVAILABLE"
    assert len(rsvqa.load_questions()) == 0

    assert vrsbench.available is False
    assert vrsbench.status_message == "NOT RUN — DATASET NOT AVAILABLE"
    assert len(vrsbench.load_annotations()) == 0


def test_preprocessing_functions():
    """Verify multi-spectral band normalization and 4-channel RGB+NIR extraction."""
    raw_data = np.random.randint(0, 10000, size=(12, 50, 50)).astype(np.float32)
    norm = normalize_raster_bands(raw_data, target_size=(120, 120))
    assert norm.shape == (12, 120, 120)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

    rgb_nir = extract_rgb_nir(norm)
    assert rgb_nir.shape == (4, 120, 120)


def test_adapted_model_unloaded_inference_fails(sample_asset):
    """Verify calling inference on untrained/unloaded adapted model raises RuntimeError."""
    adapted = AdaptedRemoteSensingModel()
    assert adapted.is_trained is False

    with pytest.raises(RuntimeError) as exc_vqa:
        adapted.answer_question(sample_asset, "Is there forest?")
    assert "TRAINING_READY" in str(exc_vqa.value)

    with pytest.raises(RuntimeError) as exc_cap:
        adapted.generate_caption(sample_asset)
    assert "TRAINING_READY" in str(exc_cap.value)


def test_adapted_model_missing_checkpoint_fails():
    """Verify attempting to load a non-existent checkpoint raises FileNotFoundError."""
    adapted = AdaptedRemoteSensingModel()
    with pytest.raises(FileNotFoundError) as exc_info:
        adapted.load_checkpoint("/fake/path/to/rs_checkpoint.pth")
    assert "REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED" in str(exc_info.value)


def test_model_registry_contains_adapted_vlm():
    """Verify ModelRegistry correctly registers adapted_remote_sensing_vlm with TRAINING_READY status."""
    models = model_registry.list_models()
    model_ids = [m.model_id for m in models]
    assert "adapted_remote_sensing_vlm" in model_ids

    adapted_info = next(m for m in models if m.model_id == "adapted_remote_sensing_vlm")
    assert adapted_info.status == "TRAINING_READY"
    assert adapted_info.remote_sensing_adapted is True
    assert adapted_info.training_dataset == "BigEarthNet-19"


def test_orchestrator_transparent_fallback_vqa(sample_asset):
    """Verify orchestrator execution trace includes ADAPTED_MODEL_UNAVAILABLE and RULE_BASED_FALLBACK_SELECTED for VQA."""
    orchestrator = PipelineOrchestrator()
    req = OrchestratorRequest(query="Is there vegetation in this scene?", t1_image_id=sample_asset.id)
    resp = orchestrator.execute(req)

    assert resp.status == "COMPLETED"
    assert resp.task == "SINGLE_IMAGE_VQA"

    trace_events = [t["event"] for t in resp.execution_trace]
    assert "ADAPTED_MODEL_UNAVAILABLE" in trace_events
    assert "RULE_BASED_FALLBACK_SELECTED" in trace_events
    assert "VQA_STARTED" in trace_events
    assert "VQA_COMPLETED" in trace_events

    # Verify fallback event details
    unavail_event = next(t for t in resp.execution_trace if t["event"] == "ADAPTED_MODEL_UNAVAILABLE")
    assert unavail_event["details"]["status"] == "TRAINING_READY"


def test_orchestrator_transparent_fallback_caption(sample_asset):
    """Verify orchestrator execution trace includes ADAPTED_MODEL_UNAVAILABLE and RULE_BASED_FALLBACK_SELECTED for Captioning."""
    orchestrator = PipelineOrchestrator()
    req = OrchestratorRequest(query="Describe this satellite image.", t1_image_id=sample_asset.id)
    resp = orchestrator.execute(req)

    assert resp.status == "COMPLETED"
    assert resp.task == "SINGLE_IMAGE_CAPTION"

    trace_events = [t["event"] for t in resp.execution_trace]
    assert "ADAPTED_MODEL_UNAVAILABLE" in trace_events
    assert "RULE_BASED_FALLBACK_SELECTED" in trace_events
    assert "CAPTION_STARTED" in trace_events
    assert "CAPTION_COMPLETED" in trace_events
