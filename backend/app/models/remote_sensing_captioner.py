from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import numpy as np

from app.config import settings
from app.schemas import RasterAsset, CaptionOutput
from app.models.land_cover_classifier import SpectralLandCoverClassifier
from app.services.raster_ingestion_service import raster_ingestion_service
import rasterio

class BaseRemoteSensingCaptioner(ABC):
    """Base interface for Remote-Sensing Image Captioning."""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @property
    def is_trained(self) -> bool:
        return False

    @property
    def adaptation_status(self) -> str:
        return "RULE_BASED_BASELINE"

    @abstractmethod
    def generate_caption(self, asset: RasterAsset) -> CaptionOutput:
        """
        Generate a comprehensive, evidence-backed description of the remote-sensing asset.
        Returns a structured CaptionOutput.
        """
        pass

class RuleBasedRemoteSensingCaptioner(BaseRemoteSensingCaptioner):
    """
    Baseline implementation of Scene Captioning.
    Combines spectral insights and raster metadata to form a grounded, safe description.
    """
    
    def __init__(self, classifier: SpectralLandCoverClassifier = None):
        self._classifier = classifier or SpectralLandCoverClassifier()
        
    @property
    def model_name(self) -> str:
        return "RuleBasedRemoteSensingCaptioner"
        
    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def is_trained(self) -> bool:
        return False

    @property
    def adaptation_status(self) -> str:
        return "RULE_BASED_BASELINE"

    def _get_pixels_and_metadata(self, asset: RasterAsset):
        file_path = Path(asset.path)
        if not file_path.is_absolute():
            file_path = settings.base_dir / asset.path
        with rasterio.open(file_path) as src:
            pixels = src.read()
            descriptions = list(src.descriptions)
            interps = [c.name for c in src.colorinterp]
            metadata = {
                "crs": str(src.crs),
                "resolution": src.res,
                "bands": src.count,
                "descriptions": descriptions,
                "modality": asset.modality
            }
        return pixels, descriptions, interps, metadata

    def generate_caption(self, asset: RasterAsset) -> CaptionOutput:
        warnings = []
        
        try:
            pixels, descs, interps, metadata = self._get_pixels_and_metadata(asset)
        except Exception as e:
            return CaptionOutput(
                caption="Failed to analyze image due to a read error.",
                confidence=0.0,
                model=self.model_name,
                warnings=[f"Data read error: {str(e)}"]
            )
            
        cls_result = self._classifier.classify_pixels(
            pixels, band_descriptions=descs, color_interpretations=interps
        )
        
        warnings.extend(cls_result.warnings)
        
        features_dict = {}
        for feat_name, stats in cls_result.features.items():
            if hasattr(stats, "get") and stats.get("mean") is not None:
                val = stats.get("mean")
            elif hasattr(stats, "mean"):
                val = stats.mean() if callable(stats.mean) else stats.mean
            else:
                try:
                    val = float(np.mean(stats))
                except Exception:
                    val = 0.0
            features_dict[feat_name.upper()] = val
            features_dict[feat_name.lower()] = val
                
        # Safe description generation avoiding hallucinated bounding boxes or locations
        caption_parts = []
        
        modality_str = "SAR" if asset.modality == "SAR" else "optical"
        
        caption_parts.append(
            f"This is a {modality_str} remote-sensing image with a primary land-cover classification of {cls_result.land_cover}."
        )
        
        if features_dict.get("NDVI", 0) > 0.3:
            caption_parts.append("The area exhibits strong signatures of healthy vegetation.")
        elif cls_result.land_cover == "VEGETATION":
            caption_parts.append("Vegetation signatures are present but weak in near-infrared intensity.")
            
        if features_dict.get("NDWI", 0) > 0.0:
            caption_parts.append("Water bodies or significant moisture are detected in the scene.")
            
        if features_dict.get("NDBI", 0) > 0.05:
            caption_parts.append("There are indications of built-up surfaces or infrastructure.")
            
        if not any(k in features_dict for k in ["NDVI", "NDWI", "NDBI"]):
            caption_parts.append("No specialized multispectral indices could be computed; description relies purely on baseline channel intensity.")
            warnings.append("Lacking multispectral bands for detailed captioning.")
            
        caption = " ".join(caption_parts)
        
        evidence = {
            "metadata": metadata,
            "spectral_features": features_dict,
            "regions": []
        }
        
        return CaptionOutput(
            caption=caption,
            confidence=cls_result.confidence,
            evidence=evidence,
            model=f"{self.model_name}_v{self.version}",
            warnings=warnings
        )
