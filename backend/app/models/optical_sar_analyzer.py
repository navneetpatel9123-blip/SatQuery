import numpy as np
import rasterio
from typing import Dict, Any, List
from abc import ABC, abstractmethod

from app.schemas import RasterAsset, OpticalSAROutput, OpticalSARRegion, AgreementScore
from app.models.land_cover_classifier import SpectralLandCoverClassifier

class BaseOpticalSARAnalyzer(ABC):
    """Abstract interface for Optical-SAR Cross-Modal Analysis."""
    
    @abstractmethod
    def analyze(self, optical_asset: RasterAsset, sar_asset: RasterAsset) -> OpticalSAROutput:
        pass

class RuleBasedOpticalSARAnalyzer(BaseOpticalSARAnalyzer):
    """Rule-based baseline for Optical-SAR fusion."""
    
    def __init__(self):
        self.optical_classifier = SpectralLandCoverClassifier()
        # Rule-based thresholds
        self.sar_high_backscatter_thresh = 0.5  # Arbitrary normalized threshold for built-up
        self.sar_low_backscatter_thresh = 0.1   # Arbitrary normalized threshold for water
        
    def _check_coregistration(self, opt: RasterAsset, sar: RasterAsset) -> str:
        crs_match = opt.crs == sar.crs
        b1, b2 = opt.bounds, sar.bounds
        overlap = (max(b1[0], b2[0]) < min(b1[2], b2[2])) and (max(b1[1], b2[1]) < min(b1[3], b2[3]))
        
        if not overlap:
            return "INCOMPATIBLE"
            
        dims_match = (opt.width == sar.width) and (opt.height == sar.height)
        transform_match = np.allclose(opt.transform, sar.transform, rtol=1e-4) if (opt.transform and sar.transform) else False
        
        if crs_match and dims_match and transform_match:
            return "CO_REGISTERED"
        elif crs_match and overlap:
            return "ALIGNMENT_REQUIRED"
            
        return "INCOMPATIBLE"

    def analyze(self, optical_asset: RasterAsset, sar_asset: RasterAsset) -> OpticalSAROutput:
        warnings = []
        
        # 1. Co-registration Check
        coreg_status = self._check_coregistration(optical_asset, sar_asset)
        if coreg_status == "INCOMPATIBLE":
            return OpticalSAROutput(
                status="FAILED",
                summary={"error": "Images are geographically incompatible and do not overlap."},
                regions=[]
            )
        
        if coreg_status == "ALIGNMENT_REQUIRED":
            warnings.append("Images overlap but require spatial alignment/resampling. Performing unaligned analysis (BASELINE).")
            
        # 2. Load Data (Windowed/Full for baseline)
        with rasterio.open(optical_asset.path) as src_opt:
            opt_data = src_opt.read()
            opt_colorinterp = src_opt.colorinterp
            opt_descriptions = src_opt.descriptions
            
        with rasterio.open(sar_asset.path) as src_sar:
            sar_data = src_sar.read()
            sar_descriptions = src_sar.descriptions
            
        # 3. Optical Features (Reuse Phase 4)
        opt_cls_res = self.optical_classifier.classify_pixels(opt_data, band_descriptions=opt_descriptions)
        
        opt_features = {}
        for feat_name, stats in opt_cls_res.features.items():
            k = feat_name.upper()
            if hasattr(stats, "get") and stats.get("mean") is not None:
                opt_features[k] = stats.get("mean")
            elif hasattr(stats, "mean"):
                opt_features[k] = stats.mean() if callable(stats.mean) else stats.mean
            else:
                try:
                    opt_features[k] = float(np.mean(stats))
                except Exception:
                    opt_features[k] = 0.0

        if not any(k in opt_features for k in ["NDVI", "NDWI", "NDBI"]):
            warnings.append("NDBI/NDVI cannot be computed from required bands. NIR unavailable.")
            
        # 4. SAR Features
        # For baseline, we flatten all bands, or use band 1
        sar_flat = sar_data[0].astype(np.float32)
        
        if sar_descriptions and sar_descriptions[0]:
            polarization = sar_descriptions[0]
        else:
            polarization = "UNKNOWN"
            
        sar_mean = float(np.mean(sar_flat))
        sar_median = float(np.median(sar_flat))
        sar_std = float(np.std(sar_flat))
        sar_min = float(np.min(sar_flat))
        sar_max = float(np.max(sar_flat))
        
        sar_features = {
            "mean_backscatter": sar_mean,
            "median_backscatter": sar_median,
            "std_backscatter": sar_std,
            "min": sar_min,
            "max": sar_max,
            "polarization": polarization
        }
        
        # 5. Cross-Modal Fusion
        # Baseline: 1 global region for the image overlap
        opt_class = opt_cls_res.land_cover
        
        agreement = AgreementScore.INSUFFICIENT_DATA
        explanation = "Insufficient data to determine cross-modal agreement."
        fusion_conf = opt_cls_res.confidence
        
        if opt_class == "BUILT_UP":
            if sar_mean > self.sar_high_backscatter_thresh:
                agreement = AgreementScore.STRONG_AGREEMENT
                explanation = f"Built-up classification is supported by elevated NDBI and compatible high SAR backscatter (mean: {sar_mean:.2f})."
                fusion_conf = min(1.0, fusion_conf + 0.1)
            else:
                agreement = AgreementScore.CONFLICT
                explanation = f"Optical indicates BUILT_UP, but SAR backscatter (mean: {sar_mean:.2f}) is too low for typical urban structures."
                fusion_conf = max(0.0, fusion_conf - 0.3)
                warnings.append("Conflict detected for Built-Up.")
                
        elif opt_class == "WATER":
            if sar_mean < self.sar_low_backscatter_thresh:
                agreement = AgreementScore.STRONG_AGREEMENT
                explanation = f"Water classification is supported by NDWI and compatible low SAR backscatter (mean: {sar_mean:.2f})."
                fusion_conf = min(1.0, fusion_conf + 0.1)
            else:
                agreement = AgreementScore.CONFLICT
                explanation = f"Optical indicates WATER, but SAR backscatter (mean: {sar_mean:.2f}) is unusually high."
                fusion_conf = max(0.0, fusion_conf - 0.3)
                warnings.append("Conflict detected for Water.")
                
        else:
            agreement = AgreementScore.OPTICAL_DOMINANT
            explanation = f"Classification ({opt_class}) relies primarily on optical evidence. SAR baseline provides moderate supplementary statistics."
            
        warnings.extend(opt_cls_res.warnings)
            
        region = OpticalSARRegion(
            region_id="R001",
            classification=opt_class,
            optical_evidence=opt_features,
            sar_evidence=sar_features,
            agreement={"label": agreement.value, "score": fusion_conf},
            confidence=fusion_conf,
            explanation=explanation,
            warnings=list(set(warnings))
        )
        
        summary = {
            "coregistration": coreg_status,
            "total_regions": 1
        }
        
        return OpticalSAROutput(
            status="COMPLETED",
            regions=[region],
            summary=summary,
            execution_trace=[]
        )
