import pytest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from datetime import datetime

from app.config import settings
from app.services.raster_ingestion_service import raster_ingestion_service
from app.schemas import RasterAsset, PairValidationResult


def create_synth_tiff(
    path: Path,
    bands: int = 3,
    width: int = 10,
    height: int = 10,
    crs: str = "EPSG:4326",
    transform=None,
    dtype: str = "uint8",
    nodata: float = None,
    tags: dict = None
):
    """Helper to create a valid synthetic GeoTIFF for testing."""
    if transform is None and crs:
        transform = from_origin(-122.5, 37.8, 0.001, 0.001)
        
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        for i in range(1, bands + 1):
            dst.write(np.ones((height, width), dtype=dtype) * i, i)
        if tags:
            dst.update_tags(**tags)


# 1. Test scenario: Valid GeoTIFF (.tif)
def test_valid_geotiff(tmp_path):
    file_path = tmp_path / "valid.tif"
    create_synth_tiff(file_path)
    
    asset = raster_ingestion_service.process_and_validate("test-1", file_path, "valid.tif")
    assert asset.validation_status == "PASS"
    assert asset.filename == "valid.tif"
    assert len(asset.errors) == 0


# 2. Test scenario: Valid TIFF (.tiff)
def test_valid_tiff(tmp_path):
    file_path = tmp_path / "valid.tiff"
    create_synth_tiff(file_path)
    
    asset = raster_ingestion_service.process_and_validate("test-2", file_path, "valid.tiff")
    assert asset.validation_status == "PASS"
    assert asset.filename == "valid.tiff"


# 3. Test scenario: Invalid file (e.g. Text file renamed to .tif)
def test_invalid_file(tmp_path):
    file_path = tmp_path / "fake.tif"
    with open(file_path, "w") as f:
        f.write("This is just a random text file, not a TIFF.")
        
    asset = raster_ingestion_service.process_and_validate("test-3", file_path, "fake.tif")
    assert asset.validation_status == "FAIL"
    assert any("Unable to read this raster" in err for err in asset.errors)


# 4. Test scenario: Corrupt raster (broken binary headers)
def test_corrupt_raster(tmp_path):
    file_path = tmp_path / "corrupt.tif"
    with open(file_path, "wb") as f:
        f.write(b"II*\x00\x08\x00\x00\x00INVALID_DATA_CORRUPTION_BLAH_BLAH")
        
    asset = raster_ingestion_service.process_and_validate("test-4", file_path, "corrupt.tif")
    assert asset.validation_status == "FAIL"
    assert len(asset.errors) > 0


# 5. Test scenario: Missing CRS
def test_missing_crs(tmp_path):
    file_path = tmp_path / "no_crs.tif"
    create_synth_tiff(file_path, crs=None, transform=None)
    
    asset = raster_ingestion_service.process_and_validate("test-5", file_path, "no_crs.tif")
    # Missing CRS is a Warning, not a Fail
    assert asset.validation_status == "WARNING"
    assert any("coordinate reference system" in w for w in asset.warnings)


# 6. Test scenario: Valid CRS
def test_valid_crs(tmp_path):
    file_path = tmp_path / "with_crs.tif"
    create_synth_tiff(file_path, crs="EPSG:4326")
    
    asset = raster_ingestion_service.process_and_validate("test-6", file_path, "with_crs.tif")
    assert asset.crs == "EPSG:4326"
    assert not any("coordinate reference system" in w for w in asset.warnings)


# 7. Test scenario: Metadata extraction
def test_metadata_extraction(tmp_path):
    file_path = tmp_path / "meta_test.tif"
    tags = {"TIFFTAG_DATETIME": "2026:08:31 12:00:00"}
    create_synth_tiff(file_path, bands=4, width=15, height=20, crs="EPSG:3857", tags=tags)
    
    asset = raster_ingestion_service.process_and_validate("test-7", file_path, "meta_test.tif")
    assert asset.width == 15
    assert asset.height == 20
    assert asset.bands == 4
    assert asset.crs == "EPSG:3857"
    assert asset.epsg == 3857
    assert asset.timestamp is not None
    assert asset.timestamp.year == 2026
    assert asset.timestamp.month == 8
    assert asset.timestamp.day == 31


# 8. Test scenario: Optical image detection
def test_optical_image(tmp_path):
    file_path = tmp_path / "optical.tif"
    create_synth_tiff(file_path, bands=3)
    
    asset = raster_ingestion_service.process_and_validate("test-8", file_path, "optical.tif")
    assert asset.modality == "optical"
    assert asset.modality_certain is True


# 9. Test scenario: SAR image detection
def test_sar_image(tmp_path):
    file_path = tmp_path / "sentinel1_sar.tif"
    create_synth_tiff(file_path, bands=1)
    
    asset = raster_ingestion_service.process_and_validate("test-9", file_path, "sentinel1_sar.tif")
    assert asset.modality == "sar"
    assert asset.modality_certain is True


# 10. Test scenario: Multispectral raster detection
def test_multispectral_raster(tmp_path):
    file_path = tmp_path / "multispectral.tif"
    create_synth_tiff(file_path, bands=8)
    
    asset = raster_ingestion_service.process_and_validate("test-10", file_path, "multispectral.tif")
    assert asset.modality == "multispectral"
    assert asset.modality_certain is True


# 11. Test scenario: Optical-SAR pair validation
def test_optical_sar_pair(tmp_path):
    opt_path = tmp_path / "opt.tif"
    sar_path = tmp_path / "sar.tif"
    create_synth_tiff(opt_path, bands=3)
    create_synth_tiff(sar_path, bands=1, tags={"TIFFTAG_DATETIME": "2026:08:31 12:00:00"})
    
    # Process assets so they save to settings.upload_dir
    opt_asset = raster_ingestion_service.process_and_validate("opt-id", opt_path, "opt.tif")
    sar_asset = raster_ingestion_service.process_and_validate("sar-id", sar_path, "sar.tif")
    
    validation = raster_ingestion_service.validate_pair("opt-id", "sar-id")
    assert validation.pair_type == "optical_sar"
    assert validation.compatible is True


# 12. Test scenario: Overlapping pair
def test_overlapping_pair(tmp_path):
    t1_path = tmp_path / "t1.tif"
    t2_path = tmp_path / "t2.tif"
    create_synth_tiff(t1_path, crs="EPSG:4326")
    create_synth_tiff(t2_path, crs="EPSG:4326")
    
    raster_ingestion_service.process_and_validate("t1-id", t1_path, "t1.tif")
    raster_ingestion_service.process_and_validate("t2-id", t2_path, "t2.tif")
    
    validation = raster_ingestion_service.validate_pair("t1-id", "t2-id")
    assert validation.spatial_overlap == 100.0
    assert validation.compatible is True


# 13. Test scenario: Non-overlapping pair
def test_non_overlapping_pair(tmp_path):
    t1_path = tmp_path / "t1_west.tif"
    t2_path = tmp_path / "t2_east.tif"
    
    # Position them far apart
    trans_t1 = from_origin(-122.5, 37.8, 0.001, 0.001)
    trans_t2 = from_origin(-100.5, 37.8, 0.001, 0.001)
    
    create_synth_tiff(t1_path, crs="EPSG:4326", transform=trans_t1)
    create_synth_tiff(t2_path, crs="EPSG:4326", transform=trans_t2)
    
    raster_ingestion_service.process_and_validate("t1-west-id", t1_path, "t1_west.tif")
    raster_ingestion_service.process_and_validate("t2-east-id", t2_path, "t2_east.tif")
    
    validation = raster_ingestion_service.validate_pair("t1-west-id", "t2-east-id")
    assert validation.spatial_overlap == 0.0
    assert validation.compatible is False
    assert any("No spatial overlap" in err for err in validation.errors)


# 14. Test scenario: Incompatible CRS
def test_incompatible_crs(tmp_path):
    t1_path = tmp_path / "t1_crs4326.tif"
    t2_path = tmp_path / "t2_crs3857.tif"
    create_synth_tiff(t1_path, crs="EPSG:4326")
    create_synth_tiff(t2_path, crs="EPSG:3857")
    
    raster_ingestion_service.process_and_validate("t1-crs1", t1_path, "t1_crs4326.tif")
    raster_ingestion_service.process_and_validate("t2-crs2", t2_path, "t2_crs3857.tif")
    
    validation = raster_ingestion_service.validate_pair("t1-crs1", "t2-crs2")
    assert validation.crs_compatible is False
    assert validation.compatible is False
    assert any("CRS mismatch" in err for err in validation.errors)


# 15. Test scenario: Incompatible resolution
def test_incompatible_resolution(tmp_path):
    t1_path = tmp_path / "t1_high_res.tif"
    t2_path = tmp_path / "t2_low_res.tif"
    
    trans_t1 = from_origin(-122.5, 37.8, 0.001, 0.001)
    trans_t2 = from_origin(-122.5, 37.8, 0.005, 0.005) # 5x larger resolution
    
    create_synth_tiff(t1_path, crs="EPSG:4326", transform=trans_t1)
    create_synth_tiff(t2_path, crs="EPSG:4326", transform=trans_t2)
    
    raster_ingestion_service.process_and_validate("t1-high", t1_path, "t1_high_res.tif")
    raster_ingestion_service.process_and_validate("t2-low", t2_path, "t2_low_res.tif")
    
    validation = raster_ingestion_service.validate_pair("t1-high", "t2-low")
    assert validation.resolution_compatible is False
    # Resolution mismatch is a Warning, pair is still "compatible" in terms of overlapping bounds
    assert validation.compatible is True
    assert any("Resolution mismatch" in w for w in validation.warnings)


# 16. Test scenario: T1/T2 pair temporal order checking
def test_t1_t2_pair(tmp_path):
    t1_path = tmp_path / "t1_opt.tif"
    t2_path = tmp_path / "t2_opt.tif"
    
    tags1 = {"TIFFTAG_DATETIME": "2026:08:30 12:00:00"}
    tags2 = {"TIFFTAG_DATETIME": "2026:08:31 12:00:00"}
    
    create_synth_tiff(t1_path, crs="EPSG:4326", tags=tags1)
    create_synth_tiff(t2_path, crs="EPSG:4326", tags=tags2)
    
    raster_ingestion_service.process_and_validate("t1-opt-id", t1_path, "t1_opt.tif")
    raster_ingestion_service.process_and_validate("t2-opt-id", t2_path, "t2_opt.tif")
    
    validation = raster_ingestion_service.validate_pair("t1-opt-id", "t2-opt-id")
    assert validation.pair_type == "t1_t2"
    assert validation.temporal_valid is True
    assert validation.compatible is True


# 17. Test scenario: Invalid temporal order
def test_invalid_temporal_order(tmp_path):
    t1_path = tmp_path / "t1_late.tif"
    t2_path = tmp_path / "t2_early.tif"
    
    tags1 = {"TIFFTAG_DATETIME": "2026:08:31 12:00:00"}
    tags2 = {"TIFFTAG_DATETIME": "2026:08:30 12:00:00"} # T2 is earlier!
    
    create_synth_tiff(t1_path, crs="EPSG:4326", tags=tags1)
    create_synth_tiff(t2_path, crs="EPSG:4326", tags=tags2)
    
    raster_ingestion_service.process_and_validate("t1-late", t1_path, "t1_late.tif")
    raster_ingestion_service.process_and_validate("t2-early", t2_path, "t2_early.tif")
    
    validation = raster_ingestion_service.validate_pair("t1-late", "t2-early")
    assert validation.temporal_valid is False
    assert validation.compatible is False
    assert any("Temporal order invalid" in err for err in validation.errors)


# 18. Test scenario: Large raster preview generation
def test_large_raster_preview(tmp_path):
    file_path = tmp_path / "large_mock.tif"
    # Create somewhat larger raster to test downsampled aspect ratio calculations
    create_synth_tiff(file_path, bands=3, width=200, height=100)
    
    asset = raster_ingestion_service.process_and_validate("large-id", file_path, "large_mock.tif")
    preview_file = raster_ingestion_service.generate_preview("large-id")
    
    assert preview_file.exists()
    assert preview_file.suffix == ".png"
    
    # Read generated preview image and verify it is downsampled
    from PIL import Image
    with Image.open(preview_file) as img:
        assert img.width == 512
        assert img.height == 256 # 2:1 aspect ratio preserved!


# 19. Test scenario: Unsupported extension
def test_unsupported_extension(tmp_path):
    file_path = tmp_path / "document.pdf"
    with open(file_path, "w") as f:
        f.write("%PDF-1.4...")
        
    asset = raster_ingestion_service.process_and_validate("test-19", file_path, "document.pdf")
    assert asset.validation_status == "FAIL"
    assert any("Unsupported file extension" in err for err in asset.errors)


# 20. Test scenario: Upload size violation
def test_upload_size_violation(tmp_path, monkeypatch):
    file_path = tmp_path / "small_but_violating.tif"
    create_synth_tiff(file_path)
    
    # Set size limit to extremely low value (e.g. 0 MB) to force size limit violation
    monkeypatch.setattr(settings, "max_file_size_mb", 0)
    
    asset = raster_ingestion_service.process_and_validate("test-20", file_path, "small_but_violating.tif")
    assert asset.validation_status == "FAIL"
    assert any("Upload size limit violation" in err for err in asset.errors)
