import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from app.models.remote_sensing_captioner import RuleBasedRemoteSensingCaptioner
from app.services.raster_ingestion_service import RasterAsset, raster_ingestion_service

def _create_synthetic_geotiff(path: str, bands: int = 4, data=None, descriptions=None):
    if data is None:
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
        if descriptions:
            for i, d in enumerate(descriptions):
                dst.set_band_description(i + 1, d)
        elif bands == 4:
            dst.colorinterp = [
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
                rasterio.enums.ColorInterp.undefined
            ]

@pytest.fixture
def veg_asset(tmp_path):
    veg_data = np.zeros((4, 10, 10), dtype=np.float32)
    veg_data[0] = 0.1 # Red
    veg_data[3] = 0.6 # NIR (high NDVI)
    veg_path = tmp_path / "veg_cap.tif"
    _create_synthetic_geotiff(str(veg_path), data=veg_data, descriptions=["RED", "GREEN", "BLUE", "NIR"])
    asset = raster_ingestion_service.process_and_validate("veg_cap_1", veg_path, "veg_cap.tif")
    asset.modality = "multispectral"
    return asset
    
@pytest.fixture
def sar_asset(tmp_path):
    sar_data = np.zeros((1, 10, 10), dtype=np.float32)
    sar_path = tmp_path / "sar.tif"
    _create_synthetic_geotiff(str(sar_path), bands=1, data=sar_data)
    asset = raster_ingestion_service.process_and_validate("sar_1", sar_path, "sar.tif")
    asset.modality = "SAR"
    return asset

def test_caption_multispectral_vegetation(veg_asset):
    cap = RuleBasedRemoteSensingCaptioner()
    res = cap.generate_caption(veg_asset)
    
    assert res.confidence > 0.0
    assert "optical" in res.caption.lower()
    assert "vegetation" in res.caption.lower()
    assert "NDVI" in res.evidence["spectral_features"]
    assert len(res.evidence["regions"]) == 0 # Baseline does not hallucinate boxes

def test_caption_sar(sar_asset):
    cap = RuleBasedRemoteSensingCaptioner()
    res = cap.generate_caption(sar_asset)
    
    assert "SAR" in res.caption
    assert any("Lacking multispectral bands" in w for w in res.warnings)
