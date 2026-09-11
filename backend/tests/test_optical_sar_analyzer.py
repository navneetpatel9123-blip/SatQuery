import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from datetime import datetime

from app.schemas import RasterAsset, AgreementScore
from app.models.optical_sar_analyzer import RuleBasedOpticalSARAnalyzer

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
def test_assets(tmp_path):
    opt_path = tmp_path / "opt.tif"
    sar_path = tmp_path / "sar.tif"
    
    # 3-band Optical (simulating NIR, Red, Green to get NDVI/NDWI)
    # Band 1: NIR (120), Band 2: Red (40) -> NDVI = (120-40)/(120+40) = 0.5 (Vegetation)
    opt_data = np.array([
        np.full((5, 5), 120.0), # NIR
        np.full((5, 5), 40.0),  # Red
        np.full((5, 5), 60.0)   # Green
    ], dtype=np.float32)
    _create_raster(opt_path, opt_data, descriptions=["NIR", "Red", "Green"])
    
    sar_data = np.array([np.full((5, 5), 0.8)], dtype=np.float32) # High backscatter
    _create_raster(sar_path, sar_data, descriptions=["VV"])
    
    opt_asset = RasterAsset(
        id="opt1", filename="opt.tif", path=str(opt_path), modality="optical",
        modality_certain=True, width=5, height=5, bands=3, dtype="float32",
        crs="EPSG:4326", bounds=[0, -50, 50, 0], resolution=[10.0, 10.0],
        transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    
    sar_asset = RasterAsset(
        id="sar1", filename="sar.tif", path=str(sar_path), modality="sar",
        modality_certain=True, width=5, height=5, bands=1, dtype="float32",
        crs="EPSG:4326", bounds=[0, -50, 50, 0], resolution=[10.0, 10.0],
        transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    
    return opt_asset, sar_asset

def test_optical_sar_analyzer_co_registration(test_assets):
    opt, sar = test_assets
    analyzer = RuleBasedOpticalSARAnalyzer()
    
    # Perfect match
    assert analyzer._check_coregistration(opt, sar) == "CO_REGISTERED"
    
    # Mismatched bounds
    sar.bounds = [100, 100, 150, 150]
    assert analyzer._check_coregistration(opt, sar) == "INCOMPATIBLE"
    
    # Partial overlap
    sar.bounds = [25, -25, 75, 25]
    sar.transform = list(from_origin(25, 25, 10, 10))
    assert analyzer._check_coregistration(opt, sar) == "ALIGNMENT_REQUIRED"

def test_optical_sar_fusion_built_up(tmp_path):
    opt_path = tmp_path / "opt_bu.tif"
    sar_path = tmp_path / "sar_bu.tif"
    
    # NDBI = (SWIR - NIR) / (SWIR + NIR)
    # Built-up needs high NDBI. SWIR=150, NIR=50 -> (150-50)/(150+50) = 0.5
    opt_data = np.array([
        np.full((5, 5), 50.0),   # NIR
        np.full((5, 5), 40.0),   # Red
        np.full((5, 5), 150.0)   # SWIR
    ], dtype=np.float32)
    _create_raster(opt_path, opt_data, descriptions=["NIR", "Red", "SWIR"])
    
    # High SAR backscatter
    sar_data = np.array([np.full((5, 5), 0.8)], dtype=np.float32)
    _create_raster(sar_path, sar_data, descriptions=["VV"])
    
    opt = RasterAsset(
        id="opt_bu", filename="opt_bu.tif", path=str(opt_path), modality="optical", modality_certain=True,
        width=5, height=5, bands=3, dtype="float32", crs="EPSG:4326", bounds=[0, -50, 50, 0],
        resolution=[10.0, 10.0], transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    sar = RasterAsset(
        id="sar_bu", filename="sar_bu.tif", path=str(sar_path), modality="sar", modality_certain=True,
        width=5, height=5, bands=1, dtype="float32", crs="EPSG:4326", bounds=[0, -50, 50, 0],
        resolution=[10.0, 10.0], transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    
    analyzer = RuleBasedOpticalSARAnalyzer()
    out = analyzer.analyze(opt, sar)
    
    assert out.status == "COMPLETED"
    assert len(out.regions) == 1
    reg = out.regions[0]
    
    assert reg.classification == "BUILT_UP"
    assert reg.agreement["label"] == AgreementScore.STRONG_AGREEMENT.value
    assert reg.sar_evidence["mean_backscatter"] == pytest.approx(0.8, rel=1e-5)

def test_optical_sar_fusion_conflict(tmp_path):
    opt_path = tmp_path / "opt_conflict.tif"
    sar_path = tmp_path / "sar_conflict.tif"
    
    # NDBI high -> Built-up
    opt_data = np.array([
        np.full((5, 5), 50.0),   # NIR
        np.full((5, 5), 40.0),   # Red
        np.full((5, 5), 150.0)   # SWIR
    ], dtype=np.float32)
    _create_raster(opt_path, opt_data, descriptions=["NIR", "Red", "SWIR"])
    
    # VERY LOW SAR backscatter -> conflict with Built-up
    sar_data = np.array([np.full((5, 5), 0.05)], dtype=np.float32)
    _create_raster(sar_path, sar_data, descriptions=["VV"])
    
    opt = RasterAsset(
        id="opt_c", filename="opt.tif", path=str(opt_path), modality="optical", modality_certain=True,
        width=5, height=5, bands=3, dtype="float32", crs="EPSG:4326", bounds=[0, -50, 50, 0],
        resolution=[10.0, 10.0], transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    sar = RasterAsset(
        id="sar_c", filename="sar.tif", path=str(sar_path), modality="sar", modality_certain=True,
        width=5, height=5, bands=1, dtype="float32", crs="EPSG:4326", bounds=[0, -50, 50, 0],
        resolution=[10.0, 10.0], transform=list(from_origin(0, 0, 10, 10)), validation_status="PASS"
    )
    
    analyzer = RuleBasedOpticalSARAnalyzer()
    out = analyzer.analyze(opt, sar)
    
    assert out.status == "COMPLETED"
    reg = out.regions[0]
    assert reg.classification == "BUILT_UP"
    assert reg.agreement["label"] == AgreementScore.CONFLICT.value
    assert any("Conflict detected" in w for w in reg.warnings)
