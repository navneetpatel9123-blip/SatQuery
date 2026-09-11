"""
SatQuery AI — Change Understander (Phase 4 / Task 2)
=====================================================

Bi-temporal change understanding module that answers:
    WHAT changed?  FROM what?  TO what?
    WHAT TYPE of change is it?  HOW STRONG is the evidence?

This is a transparent rule-based baseline.  It is NOT a trained ML
change-understanding model.  It consumes the structured output of the
Phase 3 change-detection pipeline and applies the existing
``SpectralLandCoverClassifier`` from Phase 4 / Task 1 to each temporal
region before comparing results.

Architecture
------------
    BaseChangeUnderstander          (ABC — pluggable interface)
            ↓
    RuleBasedChangeUnderstander     (spectral + rule baseline)

Future replacements (MLChangeUnderstander, VLMChangeUnderstander,
HybridChangeUnderstander) implement ``BaseChangeUnderstander`` and
register via the same interface.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.models.land_cover_classifier import (
    ClassificationResult,
    LandCoverClass,
    SpectralBand,
    SpectralLandCoverClassifier,
)


# ─── Change Taxonomy ────────────────────────────────────────────────


class ChangeTypeClass(str, Enum):
    """Supported bi-temporal change type labels."""
    BUILT_UP_EXPANSION = "BUILT_UP_EXPANSION"
    DEMOLITION = "DEMOLITION"
    VEGETATION_GAIN = "VEGETATION_GAIN"
    DEFORESTATION = "DEFORESTATION"
    WATER_BODY_CHANGE = "WATER_BODY_CHANGE"
    AGRICULTURAL_CONVERSION = "AGRICULTURAL_CONVERSION"
    ROAD_INFRASTRUCTURE_DEVELOPMENT = "ROAD_INFRASTRUCTURE_DEVELOPMENT"
    UNKNOWN_CHANGE = "UNKNOWN_CHANGE"


# ─── Confidence Label ───────────────────────────────────────────────


class ConfidenceLabel(str, Enum):
    """Human-readable confidence tiers (NOT calibrated probability)."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ─── Data Containers ────────────────────────────────────────────────


@dataclass
class FeatureEvidence:
    """One spectral-feature comparison between T1 and T2.

    Attributes:
        feature:        Name of the spectral index or statistic.
        t1:             Value in the T1 (pre-change) image.
        t2:             Value in the T2 (post-change) image.
        delta:          Signed difference ``t2 − t1``.
        interpretation: Human-readable meaning of the delta.
    """
    feature: str = ""
    t1: float = 0.0
    t2: float = 0.0
    delta: float = 0.0
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "t1": round(self.t1, 6),
            "t2": round(self.t2, 6),
            "delta": round(self.delta, 6),
            "interpretation": self.interpretation,
        }


@dataclass
class ChangeConfidence:
    """Evidence-based confidence score (NOT calibrated probability).

    Attributes:
        score:   Aggregate score in ``[0.0, 1.0]``.
        label:   Tier label: HIGH / MEDIUM / LOW.
        factors: List of human-readable factor descriptions.
    """
    score: float = 0.0
    label: str = ConfidenceLabel.LOW.value
    factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "label": self.label,
            "factors": list(self.factors),
        }


@dataclass
class TraceEvent:
    """One operational trace event in the change-understanding pipeline.

    Attributes:
        event:       Event identifier.
        region_id:   Associated region ID (if applicable).
        timestamp:   ISO-8601 timestamp string.
        duration_ms: Wall-clock milliseconds for this step.
        details:     Freeform metadata about the event.
    """
    event: str = ""
    region_id: str = ""
    timestamp: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "region_id": self.region_id,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 3),
            "details": dict(self.details),
        }


@dataclass
class RegionChangeResult:
    """Structured result for one change region.

    Contains all information required to explain a bi-temporal transition
    for a single detected region.
    """
    region_id: str = ""
    bbox: List[float] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    pixel_area: int = 0
    geospatial_area: Optional[float] = None

    t1_land_cover: str = LandCoverClass.UNKNOWN.value
    t1_land_cover_confidence: float = 0.0
    t2_land_cover: str = LandCoverClass.UNKNOWN.value
    t2_land_cover_confidence: float = 0.0

    change_type: str = ChangeTypeClass.UNKNOWN_CHANGE.value
    change_confidence: ChangeConfidence = field(default_factory=ChangeConfidence)

    spectral_features_t1: Dict[str, Any] = field(default_factory=dict)
    spectral_features_t2: Dict[str, Any] = field(default_factory=dict)
    feature_deltas: List[FeatureEvidence] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    description: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid) if self.centroid else None,
            "pixel_area": self.pixel_area,
            "geospatial_area": self.geospatial_area,
            "t1_land_cover": self.t1_land_cover,
            "t1_land_cover_confidence": round(self.t1_land_cover_confidence, 4),
            "t2_land_cover": self.t2_land_cover,
            "t2_land_cover_confidence": round(self.t2_land_cover_confidence, 4),
            "change_type": self.change_type,
            "change_confidence": self.change_confidence.to_dict(),
            "spectral_features_t1": dict(self.spectral_features_t1),
            "spectral_features_t2": dict(self.spectral_features_t2),
            "feature_deltas": [fd.to_dict() for fd in self.feature_deltas],
            "evidence": list(self.evidence),
            "description": self.description,
            "warnings": list(self.warnings),
        }


@dataclass
class ChangeUnderstandingResult:
    """Aggregate result across all change regions.

    Attributes:
        analysis_id:  Unique identifier for this analysis run.
        regions:      Per-region change results.
        summary:      Aggregate statistics.
        execution_trace: Operational trace events.
        warnings:     Top-level warnings.
    """
    analysis_id: str = ""
    regions: List[RegionChangeResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[TraceEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "regions": [r.to_dict() for r in self.regions],
            "summary": dict(self.summary),
            "execution_trace": [t.to_dict() for t in self.execution_trace],
            "warnings": list(self.warnings),
        }


# ─── Phase 3 Input Adapter ──────────────────────────────────────────


@dataclass
class ChangeRegionInput:
    """Adapter dataclass that normalises Phase 3 ``ChangeRegion`` output
    into the fields consumed by the Change Understander.

    This decouples the understander from the Pydantic schema layer so it
    can be used directly from Python without FastAPI.

    Attributes:
        region_id:       Unique region identifier.
        bbox:            ``[west, south, east, north]`` in CRS units.
        centroid:        ``[lon, lat]`` if available.
        area_sq_m:       Geospatial area in square metres (may be 0).
        phase3_confidence: The detection-level confidence from Phase 3.
        phase3_change_type: The raw change-type string from Phase 3.
        phase3_description: The description string from Phase 3.
    """
    region_id: str = ""
    bbox: List[float] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    area_sq_m: float = 0.0
    phase3_confidence: float = 0.0
    phase3_change_type: str = ""
    phase3_description: str = ""

    @classmethod
    def from_phase3_change_region(cls, cr: Any) -> "ChangeRegionInput":
        """Create from a Phase 3 ``ChangeRegion`` Pydantic model.

        Accepts any object with the attributes: ``region_id``, ``bbox``,
        ``centroid``, ``area_sq_m``, ``confidence``, ``change_type``,
        ``description``.  This keeps the import of ``app.schemas``
        optional for stand-alone usage.
        """
        change_type_val = cr.change_type
        if hasattr(change_type_val, "value"):
            change_type_val = change_type_val.value
        return cls(
            region_id=cr.region_id,
            bbox=list(cr.bbox),
            centroid=list(cr.centroid) if cr.centroid else None,
            area_sq_m=cr.area_sq_m,
            phase3_confidence=cr.confidence,
            phase3_change_type=str(change_type_val),
            phase3_description=cr.description,
        )


# ─── Abstract Base ──────────────────────────────────────────────────


class BaseChangeUnderstander(ABC):
    """Pluggable interface for bi-temporal change understanding.

    Concrete implementations (rule-based, ML, VLM, hybrid) must
    implement :meth:`understand_changes`.
    """

    @abstractmethod
    def understand_changes(
        self,
        t1_pixels: np.ndarray,
        t2_pixels: np.ndarray,
        change_regions: List[ChangeRegionInput],
        *,
        band_descriptions: Optional[List[Optional[str]]] = None,
        color_interpretations: Optional[List[Optional[str]]] = None,
        band_overrides: Optional[Dict[SpectralBand, int]] = None,
        resolution: Optional[List[float]] = None,
    ) -> ChangeUnderstandingResult:
        """Classify and explain changes across detected regions.

        Parameters:
            t1_pixels:           ``(bands, height, width)`` pre-change raster.
            t2_pixels:           ``(bands, height, width)`` post-change raster.
            change_regions:      List of detected change regions.
            band_descriptions:   Per-band description strings.
            color_interpretations: Per-band colour interpretation labels.
            band_overrides:      Explicit ``{SpectralBand: index}`` map.
            resolution:          ``[x_res, y_res]`` in CRS units.

        Returns:
            A :class:`ChangeUnderstandingResult`.
        """
        ...

    @property
    @abstractmethod
    def understander_name(self) -> str:
        """Human-readable name."""
        ...

    @property
    @abstractmethod
    def understander_version(self) -> str:
        """Semver-style version string."""
        ...

    @property
    def is_trained_model(self) -> bool:  # noqa: D401
        """Whether this understander is backed by a learned model."""
        return False


# ─── Rule-Based Change Understander ─────────────────────────────────


# Transition rules: (t1_class, t2_class) → (ChangeTypeClass, rule_text)
# Tuples checked in order; first match wins.
_TRANSITION_RULES: List[
    Tuple[Optional[LandCoverClass], Optional[LandCoverClass], ChangeTypeClass, str]
] = [
    # ── Agriculture → Built-up ───────────────────────────────────
    (LandCoverClass.AGRICULTURE, LandCoverClass.BUILT_UP,
     ChangeTypeClass.BUILT_UP_EXPANSION,
     "Agricultural land transitioned to built-up area, indicating built-up expansion."),

    # ── Agriculture → Road ───────────────────────────────────────
    (LandCoverClass.AGRICULTURE, LandCoverClass.ROAD_INFRASTRUCTURE,
     ChangeTypeClass.ROAD_INFRASTRUCTURE_DEVELOPMENT,
     "Agricultural land transitioned to road infrastructure."),

    # ── Vegetation → Built-up ────────────────────────────────────
    (LandCoverClass.VEGETATION, LandCoverClass.BUILT_UP,
     ChangeTypeClass.BUILT_UP_EXPANSION,
     "Vegetated area transitioned to built-up land, indicating built-up expansion."),

    # ── Vegetation → Road ────────────────────────────────────────
    (LandCoverClass.VEGETATION, LandCoverClass.ROAD_INFRASTRUCTURE,
     ChangeTypeClass.ROAD_INFRASTRUCTURE_DEVELOPMENT,
     "Vegetated area transitioned to road infrastructure."),

    # ── Vegetation → Bare Soil ───────────────────────────────────
    (LandCoverClass.VEGETATION, LandCoverClass.BARE_SOIL,
     ChangeTypeClass.DEFORESTATION,
     "Vegetated area transitioned to bare soil, consistent with possible deforestation or vegetation loss."),

    # ── Vegetation → Agriculture ─────────────────────────────────
    (LandCoverClass.VEGETATION, LandCoverClass.AGRICULTURE,
     ChangeTypeClass.AGRICULTURAL_CONVERSION,
     "Vegetated area transitioned to agricultural land."),

    # ── Built-up → Bare Soil ────────────────────────────────────
    (LandCoverClass.BUILT_UP, LandCoverClass.BARE_SOIL,
     ChangeTypeClass.DEMOLITION,
     "Built-up area transitioned to bare soil, consistent with possible demolition."),

    # ── Agriculture → Bare Soil ──────────────────────────────────
    (LandCoverClass.AGRICULTURE, LandCoverClass.BARE_SOIL,
     ChangeTypeClass.AGRICULTURAL_CONVERSION,
     "Agricultural land transitioned to bare soil, indicating agricultural land conversion."),

    # ── Bare Soil → Vegetation ───────────────────────────────────
    (LandCoverClass.BARE_SOIL, LandCoverClass.VEGETATION,
     ChangeTypeClass.VEGETATION_GAIN,
     "Bare soil transitioned to vegetation, indicating vegetation gain."),

    # ── Agriculture → Vegetation ─────────────────────────────────
    (LandCoverClass.AGRICULTURE, LandCoverClass.VEGETATION,
     ChangeTypeClass.VEGETATION_GAIN,
     "Agricultural land transitioned to vegetation, indicating vegetation gain."),

    # ── Bare Soil → Built-up ────────────────────────────────────
    (LandCoverClass.BARE_SOIL, LandCoverClass.BUILT_UP,
     ChangeTypeClass.BUILT_UP_EXPANSION,
     "Bare soil transitioned to built-up area, indicating new construction."),

    # ── Bare Soil → Road ─────────────────────────────────────────
    (LandCoverClass.BARE_SOIL, LandCoverClass.ROAD_INFRASTRUCTURE,
     ChangeTypeClass.ROAD_INFRASTRUCTURE_DEVELOPMENT,
     "Bare soil transitioned to road infrastructure."),

    # ── Bare Soil → Agriculture ──────────────────────────────────
    (LandCoverClass.BARE_SOIL, LandCoverClass.AGRICULTURE,
     ChangeTypeClass.AGRICULTURAL_CONVERSION,
     "Bare soil transitioned to agricultural land."),
]

# Water transitions use a wildcard approach (see _classify_transition).


class RuleBasedChangeUnderstander(BaseChangeUnderstander):
    """Transparent rule-based bi-temporal change understander.

    For each detected change region this implementation:
      1. Extracts the corresponding T1 and T2 pixel windows.
      2. Classifies both with :class:`SpectralLandCoverClassifier`.
      3. Computes spectral-feature deltas.
      4. Applies deterministic transition rules.
      5. Calculates an evidence-based confidence score.
      6. Generates a natural-language description from structured data.

    This is NOT a trained ML model.
    """

    def __init__(
        self,
        classifier: Optional[SpectralLandCoverClassifier] = None,
    ) -> None:
        self._classifier = classifier or SpectralLandCoverClassifier()

    # ── Identity ─────────────────────────────────────────────────

    @property
    def understander_name(self) -> str:  # noqa: D401
        return "RuleBasedChangeUnderstander"

    @property
    def understander_version(self) -> str:  # noqa: D401
        return "1.0.0"

    @property
    def is_trained_model(self) -> bool:  # noqa: D401
        return False

    # ── Public API ───────────────────────────────────────────────

    def understand_changes(
        self,
        t1_pixels: np.ndarray,
        t2_pixels: np.ndarray,
        change_regions: List[ChangeRegionInput],
        *,
        band_descriptions: Optional[List[Optional[str]]] = None,
        color_interpretations: Optional[List[Optional[str]]] = None,
        band_overrides: Optional[Dict[SpectralBand, int]] = None,
        resolution: Optional[List[float]] = None,
    ) -> ChangeUnderstandingResult:
        """Run the full change-understanding pipeline."""
        analysis_id = str(uuid.uuid4())
        trace: List[TraceEvent] = []
        all_warnings: List[str] = []
        region_results: List[RegionChangeResult] = []

        # Validate inputs.
        if t1_pixels.ndim != 3 or t2_pixels.ndim != 3:
            all_warnings.append("Raster inputs must be 3-D (bands, height, width).")
            return ChangeUnderstandingResult(
                analysis_id=analysis_id,
                warnings=all_warnings,
                summary=self._build_summary([], all_warnings),
            )

        if t1_pixels.shape[0] != t2_pixels.shape[0]:
            all_warnings.append(
                f"Band count mismatch: T1 has {t1_pixels.shape[0]}, "
                f"T2 has {t2_pixels.shape[0]}."
            )

        # SAR guard: single-band data should not use optical indices.
        n_bands = min(t1_pixels.shape[0], t2_pixels.shape[0])
        if n_bands == 1:
            all_warnings.append(
                "Single-band (possible SAR) imagery detected. "
                "Optical spectral indices (NDVI/NDWI/NDBI) will not be computed."
            )

        # Trace: regions received.
        trace.append(self._trace(
            "CHANGE_REGIONS_RECEIVED", "",
            {"total_regions": len(change_regions)},
        ))

        cls_kwargs: Dict[str, Any] = {}
        if band_descriptions is not None:
            cls_kwargs["band_descriptions"] = band_descriptions
        if color_interpretations is not None:
            cls_kwargs["color_interpretations"] = color_interpretations
        if band_overrides is not None:
            cls_kwargs["band_overrides"] = band_overrides

        for cr in change_regions:
            result = self._process_region(
                cr, t1_pixels, t2_pixels, cls_kwargs, resolution, trace
            )
            region_results.append(result)
            all_warnings.extend(result.warnings)

        summary = self._build_summary(region_results, all_warnings)

        return ChangeUnderstandingResult(
            analysis_id=analysis_id,
            regions=region_results,
            summary=summary,
            execution_trace=trace,
            warnings=all_warnings,
        )

    # ── Per-Region Processing ────────────────────────────────────

    def _process_region(
        self,
        cr: ChangeRegionInput,
        t1_pixels: np.ndarray,
        t2_pixels: np.ndarray,
        cls_kwargs: Dict[str, Any],
        resolution: Optional[List[float]],
        trace: List[TraceEvent],
    ) -> RegionChangeResult:
        """Full workflow for a single change region."""
        rid = cr.region_id
        warnings: List[str] = []
        evidence_lines: List[str] = []

        # ── 1. Extract sub-regions ───────────────────────────────
        t0 = time.time_ns()
        t1_sub = self._extract_region(t1_pixels, cr.bbox)
        t2_sub = self._extract_region(t2_pixels, cr.bbox)
        trace.append(self._trace(
            "REGION_EXTRACTED", rid,
            {"t1_shape": list(t1_sub.shape), "t2_shape": list(t2_sub.shape)},
            start_ns=t0,
        ))

        pixel_area = t1_sub.shape[1] * t1_sub.shape[2] if t1_sub.ndim == 3 else 0
        geo_area = cr.area_sq_m if cr.area_sq_m > 0 else None

        # ── 2. Classify T1 ──────────────────────────────────────
        t0 = time.time_ns()
        t1_cls = self._classifier.classify_pixels(t1_sub, **cls_kwargs)
        trace.append(self._trace(
            "T1_CLASSIFICATION", rid,
            {"land_cover": t1_cls.land_cover, "confidence": t1_cls.confidence},
            start_ns=t0,
        ))
        evidence_lines.append(
            f"T1 classified as {t1_cls.land_cover} "
            f"(confidence={t1_cls.confidence:.4f})."
        )
        warnings.extend(
            f"[T1] {w}" for w in t1_cls.warnings
        )

        # ── 3. Classify T2 ──────────────────────────────────────
        t0 = time.time_ns()
        t2_cls = self._classifier.classify_pixels(t2_sub, **cls_kwargs)
        trace.append(self._trace(
            "T2_CLASSIFICATION", rid,
            {"land_cover": t2_cls.land_cover, "confidence": t2_cls.confidence},
            start_ns=t0,
        ))
        evidence_lines.append(
            f"T2 classified as {t2_cls.land_cover} "
            f"(confidence={t2_cls.confidence:.4f})."
        )
        warnings.extend(
            f"[T2] {w}" for w in t2_cls.warnings
        )

        # ── 4. Feature deltas ────────────────────────────────────
        t0 = time.time_ns()
        feature_deltas = self._compute_feature_deltas(
            t1_cls.features, t2_cls.features
        )
        trace.append(self._trace(
            "FEATURE_COMPARISON", rid,
            {"delta_count": len(feature_deltas)},
            start_ns=t0,
        ))
        for fd in feature_deltas:
            evidence_lines.append(
                f"{fd.feature}: {fd.interpretation} "
                f"(Δ={fd.delta:+.4f})."
            )

        # ── 5. Determine change type ────────────────────────────
        t0 = time.time_ns()
        change_type, rule_text = self._classify_transition(
            t1_cls.land_cover, t2_cls.land_cover
        )
        trace.append(self._trace(
            "CHANGE_RULE_EVALUATION", rid,
            {"t1": t1_cls.land_cover, "t2": t2_cls.land_cover,
             "change_type": change_type.value},
            start_ns=t0,
        ))

        t0 = time.time_ns()
        trace.append(self._trace(
            "CHANGE_CLASSIFICATION", rid,
            {"change_type": change_type.value, "rule": rule_text},
            start_ns=t0,
        ))
        evidence_lines.append(f"Rule: {rule_text}")

        # ── 6. Confidence ────────────────────────────────────────
        t0 = time.time_ns()
        change_conf = self._calculate_confidence(
            t1_cls, t2_cls, feature_deltas, change_type, cr
        )
        trace.append(self._trace(
            "CONFIDENCE_CALCULATION", rid,
            {"score": change_conf.score, "label": change_conf.label},
            start_ns=t0,
        ))

        # ── 7. Description ──────────────────────────────────────
        t0 = time.time_ns()
        description = self._generate_description(
            rid, t1_cls, t2_cls, change_type, change_conf
        )
        trace.append(self._trace(
            "DESCRIPTION_GENERATED", rid,
            {"length": len(description)},
            start_ns=t0,
        ))

        # ── 8. Assemble result ───────────────────────────────────
        trace.append(self._trace("RESULT_COMPLETED", rid, {}))

        return RegionChangeResult(
            region_id=rid,
            bbox=list(cr.bbox),
            centroid=list(cr.centroid) if cr.centroid else None,
            pixel_area=pixel_area,
            geospatial_area=geo_area,
            t1_land_cover=t1_cls.land_cover,
            t1_land_cover_confidence=t1_cls.confidence,
            t2_land_cover=t2_cls.land_cover,
            t2_land_cover_confidence=t2_cls.confidence,
            change_type=change_type.value,
            change_confidence=change_conf,
            spectral_features_t1=t1_cls.features,
            spectral_features_t2=t2_cls.features,
            feature_deltas=feature_deltas,
            evidence=evidence_lines,
            description=description,
            warnings=warnings,
        )

    # ── Region Extraction ────────────────────────────────────────

    @staticmethod
    def _extract_region(
        pixels: np.ndarray,
        bbox: List[float],
    ) -> np.ndarray:
        """Extract a pixel window from a ``(bands, H, W)`` array.

        ``bbox`` is ``[west, south, east, north]`` interpreted as
        normalised pixel coordinates within the array dimensions.
        When values are ≤ 1.0 they are treated as fractions of the
        image dimensions; otherwise they are treated as absolute pixel
        coordinates.  This keeps the module independent from georeferencing
        libraries.
        """
        if pixels.ndim != 3:
            return pixels

        _, h, w = pixels.shape
        west, south, east, north = bbox

        # Detect coordinate mode.
        if all(0 <= v <= 1.0 for v in bbox):
            c_start = int(west * w)
            c_end = int(east * w) or w
            r_start = int((1.0 - north) * h)
            r_end = int((1.0 - south) * h) or h
        else:
            c_start = max(0, int(west))
            c_end = min(w, int(east))
            r_start = max(0, int(south))
            r_end = min(h, int(north))

        # Clamp to image bounds.
        c_start = max(0, min(c_start, w))
        c_end = max(c_start, min(c_end, w))
        r_start = max(0, min(r_start, h))
        r_end = max(r_start, min(r_end, h))

        # Ensure at least 1×1 window.
        if c_end <= c_start:
            c_end = min(c_start + 1, w)
        if r_end <= r_start:
            r_end = min(r_start + 1, h)

        return pixels[:, r_start:r_end, c_start:c_end]

    # ── Feature Delta Computation ────────────────────────────────

    @staticmethod
    def _compute_feature_deltas(
        t1_features: Dict[str, Any],
        t2_features: Dict[str, Any],
    ) -> List[FeatureEvidence]:
        """Compare common spectral indices between T1 and T2."""
        deltas: List[FeatureEvidence] = []
        index_names = ["ndvi", "ndwi", "ndbi"]

        for idx_name in index_names:
            t1_idx = t1_features.get(idx_name)
            t2_idx = t2_features.get(idx_name)
            if t1_idx is None or t2_idx is None:
                continue
            t1_mean = t1_idx.get("mean", 0.0)
            t2_mean = t2_idx.get("mean", 0.0)
            delta = t2_mean - t1_mean

            # Generate interpretation.
            upper = idx_name.upper()
            if abs(delta) < 0.02:
                interp = f"{upper} remained stable"
            elif delta > 0:
                interp = f"{upper} increased"
            else:
                interp = f"{upper} decreased"

            deltas.append(FeatureEvidence(
                feature=upper,
                t1=t1_mean,
                t2=t2_mean,
                delta=delta,
                interpretation=interp,
            ))

        return deltas

    # ── Transition Classification ────────────────────────────────

    @staticmethod
    def _classify_transition(
        t1_label: str,
        t2_label: str,
    ) -> Tuple[ChangeTypeClass, str]:
        """Apply deterministic transition rules.

        Returns ``(ChangeTypeClass, rule_description)``.
        """
        # If labels are the same, no meaningful change detected.
        if t1_label == t2_label:
            return (
                ChangeTypeClass.UNKNOWN_CHANGE,
                f"T1 and T2 both classified as {t1_label}; "
                f"no land-cover transition detected.",
            )

        # If either is UNKNOWN, we cannot determine transition.
        if (
            t1_label == LandCoverClass.UNKNOWN.value
            or t2_label == LandCoverClass.UNKNOWN.value
        ):
            return (
                ChangeTypeClass.UNKNOWN_CHANGE,
                f"Insufficient spectral evidence to determine transition "
                f"({t1_label} → {t2_label}).",
            )

        # Water transitions (wildcard approach).
        water = LandCoverClass.WATER.value
        if t1_label != water and t2_label == water:
            return (
                ChangeTypeClass.WATER_BODY_CHANGE,
                f"Non-water ({t1_label}) transitioned to water, "
                f"indicating water body change.",
            )
        if t1_label == water and t2_label != water:
            return (
                ChangeTypeClass.WATER_BODY_CHANGE,
                f"Water transitioned to non-water ({t2_label}), "
                f"indicating water body change.",
            )

        # Explicit transition rules.
        try:
            t1_enum = LandCoverClass(t1_label)
            t2_enum = LandCoverClass(t2_label)
        except ValueError:
            return (
                ChangeTypeClass.UNKNOWN_CHANGE,
                f"Unrecognised land-cover labels ({t1_label} → {t2_label}).",
            )

        for rule_t1, rule_t2, ctype, rule_text in _TRANSITION_RULES:
            if rule_t1 == t1_enum and rule_t2 == t2_enum:
                return ctype, rule_text

        # No rule matched.
        return (
            ChangeTypeClass.UNKNOWN_CHANGE,
            f"No specific rule matched transition {t1_label} → {t2_label}.",
        )

    # ── Confidence Calculation ───────────────────────────────────

    @staticmethod
    def _calculate_confidence(
        t1_cls: ClassificationResult,
        t2_cls: ClassificationResult,
        feature_deltas: List[FeatureEvidence],
        change_type: ChangeTypeClass,
        cr: ChangeRegionInput,
    ) -> ChangeConfidence:
        """Compute evidence-based confidence score.

        Factors:
            1. T1 classification confidence.
            2. T2 classification confidence.
            3. Feature-change strength (how large are the deltas?).
            4. Phase 3 detection confidence.
            5. Rule agreement (did a specific rule match?).

        This is an evidence score, NOT a calibrated probability.
        """
        factors: List[str] = []
        weights: List[float] = []
        scores: List[float] = []

        # Factor 1: T1 classification confidence.
        factors.append(f"T1 classification confidence: {t1_cls.confidence:.4f}")
        weights.append(0.25)
        scores.append(t1_cls.confidence)

        # Factor 2: T2 classification confidence.
        factors.append(f"T2 classification confidence: {t2_cls.confidence:.4f}")
        weights.append(0.25)
        scores.append(t2_cls.confidence)

        # Factor 3: Feature-change strength.
        if feature_deltas:
            max_delta = max(abs(fd.delta) for fd in feature_deltas)
            # Normalise delta to [0, 1] (deltas > 0.5 are very strong).
            delta_score = min(max_delta / 0.5, 1.0)
            factors.append(
                f"Strongest feature delta: {max_delta:.4f} "
                f"(normalised score: {delta_score:.4f})"
            )
        else:
            delta_score = 0.0
            factors.append("No spectral index deltas available.")
        weights.append(0.20)
        scores.append(delta_score)

        # Factor 4: Phase 3 detection confidence.
        p3_conf = cr.phase3_confidence
        factors.append(f"Phase 3 detection confidence: {p3_conf:.4f}")
        weights.append(0.15)
        scores.append(p3_conf)

        # Factor 5: Rule agreement.
        if change_type != ChangeTypeClass.UNKNOWN_CHANGE:
            rule_score = 1.0
            factors.append("Specific transition rule matched.")
        else:
            rule_score = 0.1
            factors.append("No specific transition rule matched.")
        weights.append(0.15)
        scores.append(rule_score)

        # Weighted average.
        total_weight = sum(weights)
        if total_weight > 0:
            aggregate = sum(w * s for w, s in zip(weights, scores)) / total_weight
        else:
            aggregate = 0.0

        aggregate = round(min(max(aggregate, 0.0), 1.0), 4)

        # Determine label.
        if aggregate >= 0.70:
            label = ConfidenceLabel.HIGH.value
        elif aggregate >= 0.40:
            label = ConfidenceLabel.MEDIUM.value
        else:
            label = ConfidenceLabel.LOW.value

        return ChangeConfidence(score=aggregate, label=label, factors=factors)

    # ── Description Generation ───────────────────────────────────

    @staticmethod
    def _generate_description(
        region_id: str,
        t1_cls: ClassificationResult,
        t2_cls: ClassificationResult,
        change_type: ChangeTypeClass,
        confidence: ChangeConfidence,
    ) -> str:
        """Generate a deterministic natural-language description.

        Derived entirely from structured results — no LLM, no invented
        dates, coordinates, satellite names, or causal explanations.
        """
        t1_lc = t1_cls.land_cover.replace("_", " ").lower()
        t2_lc = t2_cls.land_cover.replace("_", " ").lower()
        ct = change_type.value.replace("_", " ").lower()
        conf_label = confidence.label.lower()

        if t1_cls.land_cover == t2_cls.land_cover:
            return (
                f"Region {region_id}: both T1 and T2 are classified as "
                f"{t1_lc}. No land-cover transition detected. "
                f"Classification confidence is {conf_label}."
            )

        parts = [
            f"Region {region_id} changed from {t1_lc} in T1 "
            f"to {t2_lc} in T2.",
        ]

        if change_type != ChangeTypeClass.UNKNOWN_CHANGE:
            parts.append(
                f"The transition is consistent with {ct}."
            )
        else:
            parts.append(
                "Insufficient spectral evidence to determine a specific "
                "change type."
            )

        parts.append(f"Classification confidence is {conf_label}.")

        return " ".join(parts)

    # ── Summary ──────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        regions: List[RegionChangeResult],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Build aggregate statistics across all regions."""
        change_counts: Dict[str, int] = {}
        total_pixels = 0
        total_area = 0.0
        has_valid_area = False
        confidence_scores: List[float] = []

        for r in regions:
            ct = r.change_type
            change_counts[ct] = change_counts.get(ct, 0) + 1
            total_pixels += r.pixel_area
            if r.geospatial_area is not None:
                total_area += r.geospatial_area
                has_valid_area = True
            confidence_scores.append(r.change_confidence.score)

        summary: Dict[str, Any] = {
            "total_regions": len(regions),
            "change_type_counts": change_counts,
            "total_changed_pixels": total_pixels,
        }

        if has_valid_area:
            summary["total_changed_area_sq_m"] = round(total_area, 2)

        if confidence_scores:
            summary["confidence_summary"] = {
                "mean": round(float(np.mean(confidence_scores)), 4),
                "min": round(float(np.min(confidence_scores)), 4),
                "max": round(float(np.max(confidence_scores)), 4),
            }
        else:
            summary["confidence_summary"] = {}

        summary["warning_count"] = len(warnings)

        return summary

    # ── Trace Helper ─────────────────────────────────────────────

    @staticmethod
    def _trace(
        event: str,
        region_id: str,
        details: Dict[str, Any],
        *,
        start_ns: Optional[int] = None,
    ) -> TraceEvent:
        now = datetime.now(timezone.utc)
        elapsed = 0.0
        if start_ns is not None:
            elapsed = (time.time_ns() - start_ns) / 1e6
        return TraceEvent(
            event=event,
            region_id=region_id,
            timestamp=now.isoformat(),
            duration_ms=elapsed,
            details=details,
        )
