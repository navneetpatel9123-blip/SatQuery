from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import numpy as np

from app.config import settings
from app.schemas import RasterAsset, VQAOutput
from app.models.land_cover_classifier import SpectralLandCoverClassifier
from app.services.raster_ingestion_service import raster_ingestion_service
import rasterio

class BaseRemoteSensingVQA(ABC):
    """Base interface for Remote-Sensing Visual Question Answering."""
    
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
    def answer_question(self, asset: RasterAsset, question: str) -> VQAOutput:
        """
        Answer a natural language question about the given remote-sensing asset.
        Returns a structured VQAOutput.
        """
        pass

class RuleBasedRemoteSensingVQA(BaseRemoteSensingVQA):
    """
    Baseline implementation of Remote-Sensing VQA.
    Uses spectral indices and heuristic rules rather than a neural VLM.
    Useful as a fallback or for strict deterministic domains.
    """
    
    def __init__(self, classifier: SpectralLandCoverClassifier = None):
        self._classifier = classifier or SpectralLandCoverClassifier()
        
    @property
    def model_name(self) -> str:
        return "RuleBasedRemoteSensingVQA"
        
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

    def answer_question(self, asset: RasterAsset, question: str) -> VQAOutput:
        warnings = []
        q = question.lower()
        
        try:
            pixels, descs, interps, metadata = self._get_pixels_and_metadata(asset)
        except Exception as e:
            return VQAOutput(
                answer="Could not read image data.",
                confidence=0.0,
                model=self.model_name,
                warnings=[f"Data read error: {str(e)}"]
            )
            
        # Classify image
        cls_result = self._classifier.classify_pixels(
            pixels, band_descriptions=descs, color_interpretations=interps
        )
        
        warnings.extend(cls_result.warnings)
        
        features_dict = {}
        for feat_name, stats in cls_result.features.items():
            k_lower = feat_name.lower()
            k_upper = feat_name.upper()
            if isinstance(stats, float) or isinstance(stats, int):
                val = float(stats)
            elif hasattr(stats, "get") and stats.get("mean") is not None:
                val = stats.get("mean")
            elif hasattr(stats, "mean"):
                val = stats.mean() if callable(stats.mean) else stats.mean
            else:
                try:
                    val = float(np.mean(stats))
                except Exception:
                    val = 0.0
            features_dict[k_lower] = val
            features_dict[k_upper] = val
                
        evidence = {
            "metadata": metadata,
            "spectral_features": features_dict,
            "regions": [] # Rule-based fallback does not ground specific objects
        }
        
        answer = "I cannot determine the answer with certainty."
        confidence = 0.1
        
        # 1. Counting logic
        if "how many" in q or "count" in q:
            answer = "Model cannot reliably count objects without instance segmentation."
            confidence = 1.0
            warnings.append("Counting requires object detection module, not currently loaded.")
            
        # 2. Water questions
        elif "water" in q:
            if cls_result.land_cover == "WATER":
                answer = "Yes, water is the dominant land cover feature in this image."
                confidence = cls_result.confidence
            else:
                ndwi = features_dict.get("NDWI")
                water_ratio = features_dict.get("water_ratio", 0.0)
                if (ndwi is not None and ndwi > 0.0) or water_ratio > 0.05:
                    answer = f"Yes, there is water present in this scene (covering approximately {water_ratio*100:.1f}% of the area)."
                    confidence = 0.85
                else:
                    answer = "No significant water bodies are detected in this image."
                    confidence = 0.85
                    
        # 3. Built-up / Buildings / Road Infrastructure
        elif "building" in q or "built-up" in q or "city" in q or "urban" in q or "structure" in q or "house" in q:
            built_ratio = features_dict.get("built_ratio", 0.0)
            if cls_result.land_cover in ["BUILT_UP", "ROAD_INFRASTRUCTURE"] or built_ratio > 0.10:
                answer = f"Yes, built-up structures/buildings are clearly visible (covering approximately {built_ratio*100:.1f}% of the scene)."
                confidence = max(cls_result.confidence, 0.85)
            else:
                ndbi = features_dict.get("NDBI")
                if ndbi is not None and ndbi > 0.05:
                    answer = f"There is some indication of built-up surfaces (NDBI: {ndbi:.2f})."
                    confidence = 0.70
                else:
                    answer = "No, the area does not contain major built-up or residential structures."
                    confidence = 0.80

        # 4. Roads / Infrastructure
        elif "road" in q or "street" in q or "highway" in q:
            built_ratio = features_dict.get("built_ratio", 0.0)
            if cls_result.land_cover in ["BUILT_UP", "ROAD_INFRASTRUCTURE"] or built_ratio > 0.10:
                answer = "Yes, asphalt road network and streets are visible connecting the area."
                confidence = 0.85
            else:
                answer = "No clear major paved road network is detected in this image."
                confidence = 0.75

        # 5. Vegetation / Agriculture / Fields
        elif "vegetation" in q or "forest" in q or "tree" in q or "green" in q or "agricultural" in q or "farm" in q or "field" in q or "crop" in q:
            veg_ratio = features_dict.get("veg_ratio", 0.0)
            soil_ratio = features_dict.get("soil_ratio", 0.0)
            if veg_ratio > 0.10:
                answer = f"Yes, green vegetation and canopy cover are visible (covering approx {veg_ratio*100:.1f}% of the area)."
                confidence = max(cls_result.confidence, 0.85)
            elif soil_ratio > 0.15 or cls_result.land_cover in ["VEGETATION", "AGRICULTURE"]:
                answer = f"Yes, vegetation and agricultural field plots are visible (covering approx {soil_ratio*100:.1f}% of the area)."
                confidence = max(cls_result.confidence, 0.80)
            else:
                ndvi = features_dict.get("NDVI")
                if ndvi is not None and ndvi > 0.3:
                    answer = f"Yes, there is vegetation/agricultural land present (NDVI: {ndvi:.2f})."
                    confidence = 0.75
                else:
                    answer = "No significant vegetation or agricultural plots are detected in this scene."
                    confidence = 0.80
                    
        # 6. Land cover general / Describe / Overview
        elif any(term in q for term in ["land cover", "landcover", "type", "visible", "describe", "overview", "identify", "scene"]):
            answer = f"The primary visible land cover is {cls_result.land_cover} (commercial/residential built-up area with roads, rooftops, and parking infrastructure)." if cls_result.land_cover == "BUILT_UP" else f"The primary visible land cover is {cls_result.land_cover}."
            confidence = max(cls_result.confidence, 0.85)
            
        else:
            # Check domain relevance for general questions
            rs_keywords = [
                "water", "lake", "river", "sea", "ocean", "pond", "stream", "reservoir",
                "vegetation", "forest", "tree", "plant", "green", "canopy", "crop", "farm", "field", "agriculture", "soil", "dirt",
                "building", "built-up", "house", "home", "structure", "roof", "urban", "city", "town", "suburb", "commercial", "industrial",
                "road", "street", "highway", "asphalt", "path", "bridge", "car", "vehicle", "parking",
                "land cover", "landcover", "terrain", "scene", "area", "region", "surface", "map",
                "change", "changed", "difference", "delta", "temporal", "t1", "t2",
                "describe", "caption", "overview", "summary", "visible", "identify", "detect", "see", "show", "is there", "are there", "what type", "what is", "where", "how many", "count", "highlight", "locate"
            ]
            if any(k in q for k in rs_keywords):
                answer = f"Based on spectral analysis of this remote sensing raster, the primary land cover is classified as {cls_result.land_cover}."
                confidence = max(cls_result.confidence, 0.75)
            else:
                answer = "This question is outside the scope of satellite image analysis. Please ask questions related to the remote-sensing imagery (e.g. land cover, water bodies, vegetation, buildings, roads, or changes in the image)."
                confidence = 0.0
                warnings.append("Out-of-domain query rejected.")
            
        return VQAOutput(
            answer=answer,
            confidence=confidence, # Heuristic confidence
            evidence=evidence,
            model=f"{self.model_name}_v{self.version}",
            warnings=warnings
        )
