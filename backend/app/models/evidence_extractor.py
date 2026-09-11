"""
Phase 5 — Evidence Extraction Engine

Converts Phase 3 (Change Detection) and Phase 4 (Change Understanding)
outputs into auditable spatial, statistical, and spectral evidence.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from app.models.change_understander import (
    ChangeRegionInput,
    RegionChangeResult,
    TraceEvent,
)


# ─── Evidence Data Structures ─────────────────────────────────────────


@dataclass
class SpatialEvidence:
    region_id: str = ""
    bbox: List[float] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    pixel_area: int = 0
    geospatial_area: Optional[float] = None
    percent_total_changed_area: Optional[float] = None
    dimensions: Dict[str, float] = field(default_factory=dict)
    geometry_type: str = "POLYGON"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid) if self.centroid else None,
            "pixel_area": self.pixel_area,
            "geospatial_area": self.geospatial_area,
            "percent_total_changed_area": self.percent_total_changed_area,
            "dimensions": dict(self.dimensions),
            "geometry_type": self.geometry_type,
        }


@dataclass
class SpectralFeatureEvidence:
    feature: str
    t1: Optional[float] = None
    t2: Optional[float] = None
    delta: Optional[float] = None
    relative_change: Optional[float] = None
    interpretation: str = ""
    available: bool = True
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "t1": self.t1,
            "t2": self.t2,
            "delta": self.delta,
            "relative_change": self.relative_change,
            "interpretation": self.interpretation,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass
class StatisticalEvidence:
    band_index: int
    t1_mean: float
    t2_mean: float
    delta_mean: float
    t1_std: float
    t2_std: float
    t1_min: float
    t2_min: float
    t1_max: float
    t2_max: float
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "band_index": self.band_index,
            "t1_mean": self.t1_mean,
            "t2_mean": self.t2_mean,
            "delta_mean": self.delta_mean,
            "t1_std": self.t1_std,
            "t2_std": self.t2_std,
            "t1_min": self.t1_min,
            "t2_min": self.t2_min,
            "t1_max": self.t1_max,
            "t2_max": self.t2_max,
            "interpretation": self.interpretation,
        }


@dataclass
class LandCoverEvidence:
    t1_class: str
    t1_confidence: float
    t2_class: str
    t2_confidence: float
    transition_description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t1_class": self.t1_class,
            "t1_confidence": self.t1_confidence,
            "t2_class": self.t2_class,
            "t2_confidence": self.t2_confidence,
            "transition_description": self.transition_description,
        }


@dataclass
class TemporalEvidence:
    t1_date: Optional[str] = None
    t2_date: Optional[str] = None
    days_elapsed: Optional[int] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t1_date": self.t1_date,
            "t2_date": self.t2_date,
            "days_elapsed": self.days_elapsed,
            "description": self.description,
        }


@dataclass
class VisualEvidence:
    t1_crop_window: Optional[List[int]] = None
    t2_crop_window: Optional[List[int]] = None
    change_mask_window: Optional[List[int]] = None
    bounding_box: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t1_crop_window": self.t1_crop_window,
            "t2_crop_window": self.t2_crop_window,
            "change_mask_window": self.change_mask_window,
            "bounding_box": self.bounding_box,
        }


@dataclass
class EvidenceResult:
    region_id: str
    change_type: str
    
    spatial_evidence: SpatialEvidence
    spectral_evidence: List[SpectralFeatureEvidence]
    statistical_evidence: List[StatisticalEvidence]
    land_cover_evidence: LandCoverEvidence
    temporal_evidence: TemporalEvidence
    visual_evidence: VisualEvidence
    
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    
    evidence_score: float = 0.0
    evidence_label: str = "INSUFFICIENT"
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)

    execution_trace: List[TraceEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "change_type": self.change_type,
            "spatial_evidence": self.spatial_evidence.to_dict(),
            "spectral_evidence": [e.to_dict() for e in self.spectral_evidence],
            "statistical_evidence": [e.to_dict() for e in self.statistical_evidence],
            "land_cover_evidence": self.land_cover_evidence.to_dict(),
            "temporal_evidence": self.temporal_evidence.to_dict(),
            "visual_evidence": self.visual_evidence.to_dict(),
            "evidence_summary": {
                "overall_strength": self.evidence_label,
                "score": round(self.evidence_score, 4),
                "supporting_evidence": list(self.supporting_evidence),
                "contradicting_evidence": list(self.contradicting_evidence),
                "warnings": list(self.warnings),
            },
            "explanation": self.explanation,
            "execution_trace": [t.to_dict() for t in self.execution_trace],
        }


# ─── Extractor Interfaces ─────────────────────────────────────────────


class BaseEvidenceExtractor(ABC):
    """Abstract base class for Phase 5 Evidence Extractors."""

    @abstractmethod
    def extract(
        self,
        t1_pixels: np.ndarray,
        t2_pixels: np.ndarray,
        change_regions: List[ChangeRegionInput],
        phase4_results: List[RegionChangeResult],
        t1_date: Optional[str] = None,
        t2_date: Optional[str] = None,
    ) -> List[EvidenceResult]:
        """Extract evidence for the given regions."""
        pass


class RuleBasedEvidenceExtractor(BaseEvidenceExtractor):
    """Baseline evidence extractor using deterministic rules and statistics."""

    def extract(
        self,
        t1_pixels: np.ndarray,
        t2_pixels: np.ndarray,
        change_regions: List[ChangeRegionInput],
        phase4_results: List[RegionChangeResult],
        t1_date: Optional[str] = None,
        t2_date: Optional[str] = None,
    ) -> List[EvidenceResult]:
        
        results = []
        
        # Calculate total change area for percentages.
        total_changed_area = sum(cr.area_sq_m for cr in change_regions)

        # Match Phase 3 and Phase 4 results by region_id
        p3_map = {cr.region_id: cr for cr in change_regions}
        
        _, h, w = t1_pixels.shape

        for p4_res in phase4_results:
            trace: List[TraceEvent] = []
            trace.append(self._trace("EVIDENCE_EXTRACTION_STARTED", p4_res.region_id))

            warnings: List[str] = list(p4_res.warnings)
            
            p3_res = p3_map.get(p4_res.region_id)
            if not p3_res:
                continue
                
            # 1. SPATIAL EVIDENCE
            t0 = time.time_ns()
            pct_area = None
            if total_changed_area > 0 and p3_res.area_sq_m > 0:
                pct_area = (p3_res.area_sq_m / total_changed_area) * 100.0
                
            dim_x = abs(p3_res.bbox[2] - p3_res.bbox[0])
            dim_y = abs(p3_res.bbox[3] - p3_res.bbox[1])
            
            spatial_ev = SpatialEvidence(
                region_id=p4_res.region_id,
                bbox=list(p3_res.bbox),
                centroid=list(p3_res.centroid) if p3_res.centroid else None,
                pixel_area=p4_res.pixel_area,
                geospatial_area=p3_res.area_sq_m if p3_res.area_sq_m > 0 else None,
                percent_total_changed_area=pct_area,
                dimensions={"width": dim_x, "height": dim_y},
                geometry_type="POLYGON"
            )
            if p3_res.area_sq_m <= 0:
                warnings.append("Geospatial area could not be safely calculated; falling back to pixel area.")
            trace.append(self._trace("SPATIAL_EVIDENCE_EXTRACTED", p4_res.region_id, (time.time_ns()-t0)/1e6))

            # 2. SPECTRAL EVIDENCE
            t0 = time.time_ns()
            spectral_ev = self._extract_spectral(p4_res)
            trace.append(self._trace("SPECTRAL_EVIDENCE_EXTRACTED", p4_res.region_id, (time.time_ns()-t0)/1e6))

            # 3. STATISTICAL EVIDENCE
            t0 = time.time_ns()
            stat_ev = self._extract_statistical(t1_pixels, t2_pixels, p3_res.bbox, w, h)
            trace.append(self._trace("STATISTICAL_EVIDENCE_EXTRACTED", p4_res.region_id, (time.time_ns()-t0)/1e6))

            # 4. LAND-COVER EVIDENCE
            t0 = time.time_ns()
            lc_ev = LandCoverEvidence(
                t1_class=p4_res.t1_land_cover,
                t1_confidence=p4_res.t1_land_cover_confidence,
                t2_class=p4_res.t2_land_cover,
                t2_confidence=p4_res.t2_land_cover_confidence,
                transition_description=f"Land-cover transition from {p4_res.t1_land_cover} to {p4_res.t2_land_cover}."
            )
            trace.append(self._trace("LAND_COVER_EVIDENCE_EXTRACTED", p4_res.region_id, (time.time_ns()-t0)/1e6))
            
            # 5. TEMPORAL EVIDENCE
            temp_desc = "Acquisition date unavailable."
            days = None
            if t1_date and t2_date:
                try:
                    d1 = datetime.fromisoformat(t1_date)
                    d2 = datetime.fromisoformat(t2_date)
                    days = abs((d2 - d1).days)
                    temp_desc = f"Change occurred over {days} days."
                except Exception:
                    temp_desc = "Acquisition dates present but could not be parsed."
            temporal_ev = TemporalEvidence(
                t1_date=t1_date,
                t2_date=t2_date,
                days_elapsed=days,
                description=temp_desc
            )
            
            # 6. VISUAL EVIDENCE
            # For window crops, we translate bbox back to pixel coordinates if possible.
            # We assume bbox here might be pixel coordinates if max > 1, otherwise fractional.
            bbox = p3_res.bbox
            if all(0 <= v <= 1.0 for v in bbox):
                c_start = int(bbox[0] * w)
                r_start = int(bbox[1] * h)
                c_end = int(bbox[2] * w)
                r_end = int(bbox[3] * h)
            else:
                c_start = max(0, int(bbox[0]))
                c_end = min(w, int(bbox[2]))
                r_start = max(0, int(bbox[1]))
                r_end = min(h, int(bbox[3]))
                
            window = [c_start, r_start, c_end, r_end]
            vis_ev = VisualEvidence(
                t1_crop_window=window,
                t2_crop_window=window,
                change_mask_window=window,
                bounding_box=list(bbox)
            )
            
            # 7 & 8. CONTRADICTION & STRENGTH
            t0 = time.time_ns()
            supporting, contradicting, score, label = self._evaluate_evidence(
                p4_res.change_type, spectral_ev, lc_ev, p4_res.change_confidence.score
            )
            trace.append(self._trace("CONTRADICTION_CHECKED", p4_res.region_id, (time.time_ns()-t0)/1e6))
            trace.append(self._trace("EVIDENCE_SCORE_CALCULATED", p4_res.region_id))
            
            # 9. HUMAN-READABLE EXPLANATION
            explanation = self._generate_explanation(p4_res.change_type, lc_ev, spectral_ev)
            
            trace.append(self._trace("EVIDENCE_SUMMARY_GENERATED", p4_res.region_id))
            trace.append(self._trace("EVIDENCE_EXTRACTION_COMPLETED", p4_res.region_id))
            
            if contradicting:
                warnings.append("Spectral evidence partially contradicts the proposed change classification.")
            
            results.append(EvidenceResult(
                region_id=p4_res.region_id,
                change_type=p4_res.change_type,
                spatial_evidence=spatial_ev,
                spectral_evidence=spectral_ev,
                statistical_evidence=stat_ev,
                land_cover_evidence=lc_ev,
                temporal_evidence=temporal_ev,
                visual_evidence=vis_ev,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                evidence_score=score,
                evidence_label=label,
                explanation=explanation,
                warnings=warnings,
                execution_trace=trace
            ))
            
        return results

    def _extract_spectral(self, p4_res: RegionChangeResult) -> List[SpectralFeatureEvidence]:
        ev = []
        # Target indices
        for feature in ["NDVI", "NDWI", "NDBI"]:
            # Handle both uppercase direct floats (tests) and lowercase dicts (actual pipeline)
            t1_obj = p4_res.spectral_features_t1.get(feature) or p4_res.spectral_features_t1.get(feature.lower())
            t2_obj = p4_res.spectral_features_t2.get(feature) or p4_res.spectral_features_t2.get(feature.lower())
            
            t1_val = t1_obj.get("mean") if isinstance(t1_obj, dict) else t1_obj
            t2_val = t2_obj.get("mean") if isinstance(t2_obj, dict) else t2_obj
            
            if t1_val is None or t2_val is None:
                reason = f"Required bands for {feature} unavailable"
                if feature == "NDVI" or feature == "NDWI":
                    reason = "NIR band unavailable"
                elif feature == "NDBI":
                    reason = "SWIR band unavailable"
                
                ev.append(SpectralFeatureEvidence(
                    feature=feature, available=False, reason=reason
                ))
                continue
                
            delta = t2_val - t1_val
            rel_change = None
            if abs(t1_val) > 1e-5:
                rel_change = delta / abs(t1_val)
                
            interp = ""
            if feature == "NDVI":
                interp = "vegetation indicator " + ("increased" if delta > 0.05 else "decreased" if delta < -0.05 else "remained stable")
            elif feature == "NDWI":
                interp = "water indicator " + ("increased" if delta > 0.05 else "decreased" if delta < -0.05 else "remained stable")
            elif feature == "NDBI":
                interp = "built-up indicator " + ("increased" if delta > 0.05 else "decreased" if delta < -0.05 else "remained stable")
                
            ev.append(SpectralFeatureEvidence(
                feature=feature,
                t1=round(t1_val, 4),
                t2=round(t2_val, 4),
                delta=round(delta, 4),
                relative_change=round(rel_change, 4) if rel_change is not None else None,
                interpretation=interp,
                available=True
            ))
            
        return ev

    def _extract_statistical(
        self, t1_pixels: np.ndarray, t2_pixels: np.ndarray, bbox: List[float], w: int, h: int
    ) -> List[StatisticalEvidence]:
        if all(0 <= v <= 1.0 for v in bbox):
            c_start = int(bbox[0] * w)
            r_start = int(bbox[1] * h)
            c_end = int(bbox[2] * w)
            r_end = int(bbox[3] * h)
        else:
            c_start = max(0, int(bbox[0]))
            c_end = min(w, int(bbox[2]))
            r_start = max(0, int(bbox[1]))
            r_end = min(h, int(bbox[3]))
            
        # Fallback if window is invalid
        if c_end <= c_start or r_end <= r_start:
            c_start, r_start, c_end, r_end = 0, 0, w, h
            
        t1_patch = t1_pixels[:, r_start:r_end, c_start:c_end]
        t2_patch = t2_pixels[:, r_start:r_end, c_start:c_end]
        
        stats = []
        bands = min(t1_patch.shape[0], t2_patch.shape[0])
        for b in range(bands):
            t1_b = t1_patch[b]
            t2_b = t2_patch[b]
            if t1_b.size == 0 or t2_b.size == 0:
                continue
                
            t1_mean = float(np.mean(t1_b))
            t2_mean = float(np.mean(t2_b))
            delta = t2_mean - t1_mean
            
            interp = f"Band {b+1} average " + ("increased" if delta > 0 else "decreased" if delta < 0 else "unchanged")
            
            stats.append(StatisticalEvidence(
                band_index=b,
                t1_mean=round(t1_mean, 4),
                t2_mean=round(t2_mean, 4),
                delta_mean=round(delta, 4),
                t1_std=round(float(np.std(t1_b)), 4),
                t2_std=round(float(np.std(t2_b)), 4),
                t1_min=round(float(np.min(t1_b)), 4),
                t2_min=round(float(np.min(t2_b)), 4),
                t1_max=round(float(np.max(t1_b)), 4),
                t2_max=round(float(np.max(t2_b)), 4),
                interpretation=interp
            ))
        return stats

    def _evaluate_evidence(
        self,
        change_type: str,
        spectral_ev: List[SpectralFeatureEvidence],
        lc_ev: LandCoverEvidence,
        p4_score: float
    ) -> tuple[List[str], List[str], float, str]:
        
        supporting = []
        contradicting = []
        
        supporting.append(f"T1 classified as {lc_ev.t1_class}")
        supporting.append(f"T2 classified as {lc_ev.t2_class}")
        
        ndvi = next((s for s in spectral_ev if s.feature == "NDVI"), None)
        ndwi = next((s for s in spectral_ev if s.feature == "NDWI"), None)
        ndbi = next((s for s in spectral_ev if s.feature == "NDBI"), None)
        
        if change_type == "BUILT_UP_EXPANSION":
            if ndbi and ndbi.available:
                if ndbi.delta and ndbi.delta > 0.05:
                    supporting.append(f"NDBI increased by {ndbi.delta}")
                elif ndbi.delta and ndbi.delta < -0.05:
                    contradicting.append(f"NDBI decreased by {abs(ndbi.delta)}")
            if ndvi and ndvi.available and ndvi.delta and ndvi.delta < -0.05:
                supporting.append(f"NDVI decreased by {abs(ndvi.delta)}")
                
        elif change_type == "DEFORESTATION" or change_type == "VEGETATION_LOSS":
            if ndvi and ndvi.available:
                if ndvi.delta and ndvi.delta < -0.05:
                    supporting.append(f"NDVI decreased by {abs(ndvi.delta)}")
                elif ndvi.delta and ndvi.delta > 0.05:
                    contradicting.append(f"NDVI increased by {ndvi.delta}")
                    
        elif change_type == "WATER_BODY_CHANGE" or "WATER" in change_type:
            if ndwi and ndwi.available:
                if ndwi.delta and abs(ndwi.delta) > 0.05:
                    direction = "increased" if ndwi.delta > 0 else "decreased"
                    supporting.append(f"NDWI {direction} by {abs(ndwi.delta)}")
                else:
                    contradicting.append("NDWI remained relatively stable")
                    
        elif change_type == "AGRICULTURAL_CONVERSION":
            if lc_ev.t1_class == "VEGETATION" or lc_ev.t1_class == "AGRICULTURE":
                supporting.append("Initial state was vegetation/agriculture")
            else:
                contradicting.append("Initial state was not vegetation/agriculture")
                
        # Base score on Phase 4 confidence, penalty for contradictions
        score = p4_score
        if contradicting:
            score = max(0.0, score - (0.2 * len(contradicting)))
        else:
            score = min(1.0, score + 0.05)
            
        if score > 0.8:
            label = "HIGH"
        elif score > 0.5:
            label = "MEDIUM"
        elif score > 0.2:
            label = "LOW"
        else:
            label = "INSUFFICIENT"
            
        return supporting, contradicting, score, label

    def _generate_explanation(
        self,
        change_type: str,
        lc_ev: LandCoverEvidence,
        spectral_ev: List[SpectralFeatureEvidence]
    ) -> str:
        
        ndvi = next((s for s in spectral_ev if s.feature == "NDVI"), None)
        ndwi = next((s for s in spectral_ev if s.feature == "NDWI"), None)
        ndbi = next((s for s in spectral_ev if s.feature == "NDBI"), None)
        
        base = f"The classification of {change_type} is supported by a transition from {lc_ev.t1_class.lower()} to {lc_ev.t2_class.lower()}."
        
        if change_type == "BUILT_UP_EXPANSION":
            if ndbi and ndbi.available and ndbi.delta and ndbi.delta > 0.05:
                return base + " This is confirmed by an increase in the built-up spectral indicator (NDBI)."
            
        if change_type == "DEFORESTATION" or change_type == "VEGETATION_LOSS":
            if ndvi and ndvi.available and ndvi.delta and ndvi.delta < -0.05:
                return base + " This is confirmed by a decrease in the vegetation spectral indicator (NDVI)."
                
        if change_type == "WATER_BODY_CHANGE" or "WATER" in change_type:
            if ndwi and ndwi.available and ndwi.delta:
                direction = "an increase" if ndwi.delta > 0 else "a decrease"
                return base + f" This is confirmed by {direction} in the water spectral indicator (NDWI)."
                
        return base

    @staticmethod
    def _trace(event: str, region_id: str, duration: float = 0.0) -> TraceEvent:
        return TraceEvent(
            event=event,
            region_id=region_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration
        )
