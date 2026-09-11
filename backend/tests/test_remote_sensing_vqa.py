import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from app.models.remote_sensing_vqa import RuleBasedRemoteSensingVQA
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
def vqa_assets(tmp_path):
    # RGB Water
    water_data = np.zeros((4, 10, 10), dtype=np.float32)
    water_data[2] = 0.5 # Blue
    water_data[3] = 0.1 # NIR
    water_path = tmp_path / "water.tif"
    _create_synthetic_geotiff(str(water_path), data=water_data, descriptions=["RED", "GREEN", "BLUE", "NIR"])
    water_asset = raster_ingestion_service.process_and_validate("water1", water_path, "water.tif")
    water_asset.modality = "optical"
    
    # Vegetation
    veg_data = np.zeros((4, 10, 10), dtype=np.float32)
    veg_data[0] = 0.1 # Red
    veg_data[3] = 0.6 # NIR (high NDVI)
    veg_path = tmp_path / "veg.tif"
    _create_synthetic_geotiff(str(veg_path), data=veg_data, descriptions=["RED", "GREEN", "BLUE", "NIR"])
    veg_asset = raster_ingestion_service.process_and_validate("veg1", veg_path, "veg.tif")
    veg_asset.modality = "multispectral"
    
    return water_asset, veg_asset

def test_vqa_water_question(vqa_assets):
    water_asset, _ = vqa_assets
    vqa = RuleBasedRemoteSensingVQA()
    
    res = vqa.answer_question(water_asset, "Is there water here?")
    
    assert res.confidence > 0.0
    assert "water" in res.answer.lower()
    assert isinstance(res.evidence, dict)

def test_vqa_vegetation_question(vqa_assets):
    _, veg_asset = vqa_assets
    vqa = RuleBasedRemoteSensingVQA()
    
    res = vqa.answer_question(veg_asset, "Are there trees or vegetation?")
    
    assert res.confidence > 0.0
    assert "vegetation" in res.answer.lower()
    assert "NDVI" in res.evidence["spectral_features"]
    
def test_vqa_counting_fallback(vqa_assets):
    water_asset, _ = vqa_assets
    vqa = RuleBasedRemoteSensingVQA()
    
    res = vqa.answer_question(water_asset, "How many buildings are there?")
    assert res.confidence == 1.0 # Heuristic override
    assert "cannot reliably count" in res.answer.lower()
    assert any("Counting requires object detection" in w for w in res.warnings)
