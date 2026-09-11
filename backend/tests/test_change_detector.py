"""
SatQuery AI — Change Detector Unit & Integration Tests
======================================================
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.models.change_detector import (
    BaseChangeDetector,
    SiameseChangeDetector,
    SpectralDiffChangeDetector,
    extract_connected_regions,
    change_detector,
)
from app.schemas import ChangeRegion, ChangeType, RasterAsset
from app.training.config import BigEarthNetConfig, ChangeDetectionTrainingConfig, validate_checkpoint_file
from app.training.change_dataset import compute_change_metrics, CDVQADatasetAdapter
from app.services.pipeline_orchestrator import orchestrator, OrchestratorRequest
from app.config import settings


def _create_synthetic_geotiff(path: str, bands: list[np.ndarray], transform=None):
    num_bands = len(bands)
    height, width = bands[0].shape
    tform = transform or from_origin(10.0, 10.0, 1.0, 1.0)
    with rasterio.open(
        path, 'w', driver='GTiff', height=height, width=width,
        count=num_bands, dtype=bands[0].dtype, crs='EPSG:4326', transform=tform
    ) as dst:
        for i, (band, desc) in enumerate(zip(bands, ["red", "green", "blue", "nir", "swir"]), start=1):
            dst.write(band, i)
            dst.set_band_description(i, desc)


def test_base_change_detector_interface():
    detector = SpectralDiffChangeDetector()
    assert isinstance(detector, BaseChangeDetector)
    assert detector.model_name == "SpectralDiffChangeDetector"
    assert detector.version == "1.0.0"
    assert detector.is_trained is True
    assert detector.adaptation_status == "RULE_BASED_BASELINE"


def test_siamese_change_detector_status():
    detector = SiameseChangeDetector()
    assert isinstance(detector, BaseChangeDetector)
    assert detector.model_name == "SiameseChangeDetector"
    assert detector.version == "1.0.0"
    assert detector.is_trained is False
    assert detector.adaptation_status == "TRAINING_READY"

    # Direct inference without checkpoint raises RuntimeError
    with pytest.raises(RuntimeError, match="TRAINING_READY"):
        detector.detect_from_arrays(np.zeros((4, 64, 64)), np.zeros((4, 64, 64)))


def test_spectral_diff_no_change():
    detector = SpectralDiffChangeDetector(sensitivity=2.0)
    # Identical arrays
    t1 = np.ones((4, 32, 32), dtype=np.float32) * 50.0
    t2 = np.ones((4, 32, 32), dtype=np.float32) * 50.0

    res = detector.detect_from_arrays(t1, t2)
    assert len(res.regions) == 0
    assert res.metadata["changed_pixels"] == 0
    assert res.metadata["change_percentage"] == 0.0


def test_spectral_diff_with_local_change():
    detector = SpectralDiffChangeDetector(sensitivity=1.0, min_region_pixels=4)
    # Uniform background
    t1 = np.ones((4, 40, 40), dtype=np.float32) * 50.0
    t2 = np.ones((4, 40, 40), dtype=np.float32) * 50.0

    # Introduce a 6x6 patch anomaly in T2 (e.g. built-up expansion)
    t2[:, 10:16, 10:16] = 500.0

    res = detector.detect_from_arrays(t1, t2)
    assert len(res.regions) >= 1
    assert res.metadata["changed_pixels"] >= 36
    assert res.metadata["change_percentage"] > 0.0

    # Check first region properties
    r = res.regions[0]
    assert isinstance(r, ChangeRegion)
    assert r.area_sq_m > 0
    assert len(r.bbox) == 4
    assert len(r.centroid) == 2
    assert 0.0 <= r.confidence <= 1.0


def test_connected_region_extraction():
    mask = np.zeros((30, 30), dtype=bool)
    # Create two disconnected regions
    mask[2:6, 2:6] = True    # 16 pixels
    mask[20:25, 20:25] = True # 25 pixels

    conf = np.ones((30, 30), dtype=np.float32) * 0.9
    transform = from_origin(100.0, 50.0, 10.0, 10.0)

    regions = extract_connected_regions(
        binary_mask=mask,
        confidence_map=conf,
        transform=transform,
        resolution=[10.0, 10.0],
        min_pixels=4,
    )

    assert len(regions) == 2
    # Sorted descending by size
    assert regions[0].area_sq_m == 25 * 100.0  # 2500 m²
    assert regions[1].area_sq_m == 16 * 100.0  # 1600 m²


def test_compute_change_metrics():
    gt = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    pred = np.array([[0.1, 0.9, 0.8], [0.2, 0.7, 0.1], [0.1, 0.1, 0.2]], dtype=np.float32)

    metrics = compute_change_metrics(pred, gt, threshold=0.5)
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["iou"] == 1.0


def test_cdvqa_adapter_missing_dataset():
    adapter = CDVQADatasetAdapter("./non_existent_path")
    assert adapter.available is False
    assert "NOT RUN" in adapter.status_message
    assert adapter.load_qa_pairs() == []


def test_training_config_validation():
    cfg = BigEarthNetConfig(epochs=5, batch_size=16, learning_rate=1e-4)
    assert cfg.validate() == []

    invalid_cfg = BigEarthNetConfig(epochs=-1, batch_size=0, learning_rate=-0.5)
    assert len(invalid_cfg.validate()) >= 3


def test_checkpoint_validator():
    res = validate_checkpoint_file("./non_existent_weights.pth")
    assert res["valid"] is False
    assert res["status"] == "NOT_FOUND"


@pytest.fixture
def change_geotiffs(tmp_path):
    settings.upload_dir = tmp_path / "uploads"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    t1_path = settings.upload_dir / "t1_cd.tif"
    t2_path = settings.upload_dir / "t2_cd.tif"

    # T1: Vegetation
    t1_bands = [
        np.full((16, 16), 30.0, dtype=np.float32),
        np.full((16, 16), 60.0, dtype=np.float32),
        np.full((16, 16), 40.0, dtype=np.float32),
        np.full((16, 16), 200.0, dtype=np.float32),
        np.full((16, 16), 50.0, dtype=np.float32),
    ]
    # T2: Center converted to Built-Up
    t2_bands = [
        np.full((16, 16), 30.0, dtype=np.float32),
        np.full((16, 16), 60.0, dtype=np.float32),
        np.full((16, 16), 40.0, dtype=np.float32),
        np.full((16, 16), 200.0, dtype=np.float32),
        np.full((16, 16), 50.0, dtype=np.float32),
    ]
    t2_bands[0][4:12, 4:12] = 120.0  # Red up
    t2_bands[3][4:12, 4:12] = 90.0   # NIR down

    _create_synthetic_geotiff(str(t1_path), t1_bands)
    _create_synthetic_geotiff(str(t2_path), t2_bands)

    from app.services.raster_ingestion_service import raster_ingestion_service
    a1 = raster_ingestion_service.process_and_validate("t1_cd_id", t1_path, "t1_cd.tif")
    a2 = raster_ingestion_service.process_and_validate("t2_cd_id", t2_path, "t2_cd.tif")

    return a1, a2


def test_end_to_end_change_detection_orchestration(change_geotiffs):
    a1, a2 = change_geotiffs

    req = OrchestratorRequest(
        query="What changed between these two dates and why?",
        t1_image_id=a1.id,
        t2_image_id=a2.id
    )

    resp = orchestrator.execute(req)
    assert resp.status == "COMPLETED"
    assert resp.task == "BI_TEMPORAL_EVIDENCE"
    assert resp.workflow == ["CHANGE_DETECTION", "CHANGE_UNDERSTANDING", "EVIDENCE_EXTRACTION"]

    # Check models used includes change detector
    model_ids = [m["model_id"] for m in resp.models_used]
    assert any(m in model_ids for m in ["siamese_change_detector", "spectral_diff_change_detector"])

    # Check trace includes change detection events
    events = [t["event"] for t in resp.execution_trace]
    assert "CHANGE_DETECTION_STARTED" in events
    assert "CHANGE_MASK_COMPUTED" in events
    assert "CHANGE_REGIONS_EXTRACTED" in events
    assert "CHANGE_DETECTION_COMPLETED" in events
    assert "PHASE_4_STARTED" in events
    assert "PHASE_5_STARTED" in events
    assert "ORCHESTRATION_COMPLETED" in events

    # Check regions
    assert len(resp.results["regions"]) >= 1
    reg = resp.results["regions"][0]
    assert "evidence" in reg
    assert reg.get("geospatial_area", 0) > 0 or reg.get("pixel_area", 0) > 0
    assert reg["change_type"] is not None
