"""
SatQuery AI — Land Cover Classifier (Phase 4 / Task 1)
======================================================

Transparent spectral-index baseline for land-cover classification.
This is NOT a trained AI model. It applies deterministic spectral-index
thresholds to raster pixel data and returns evidence-based confidence.

Architecture
------------
    BaseLandCoverClassifier   (ABC — pluggable interface)
            ↓
    SpectralLandCoverClassifier  (spectral-index baseline)

Future replacements (ResNet, Transformer, foundation model) implement
``BaseLandCoverClassifier`` and register via the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─── Land Cover Classes ─────────────────────────────────────────────


class LandCoverClass(str, Enum):
    """Supported land-cover labels."""
    WATER = "WATER"
    VEGETATION = "VEGETATION"
    AGRICULTURE = "AGRICULTURE"
    BUILT_UP = "BUILT_UP"
    BARE_SOIL = "BARE_SOIL"
    ROAD_INFRASTRUCTURE = "ROAD_INFRASTRUCTURE"
    UNKNOWN = "UNKNOWN"


# ─── Spectral Band Names ────────────────────────────────────────────


class SpectralBand(str, Enum):
    """Canonical spectral band identifiers used by the classifier."""
    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"
    NIR = "NIR"
    SWIR = "SWIR"


# ─── Data Containers ────────────────────────────────────────────────


@dataclass
class RegionalStats:
    """Robust statistics for a single-band region.

    Attributes:
        mean:   Arithmetic mean of pixel values.
        median: Median pixel value.
        std:    Standard deviation of pixel values.
        min:    Minimum pixel value.
        max:    Maximum pixel value.
        p10:    10th-percentile pixel value.
        p25:    25th-percentile pixel value.
        p75:    75th-percentile pixel value.
        p90:    90th-percentile pixel value.
    """
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0


@dataclass
class ClassificationResult:
    """Structured output of a land-cover classification.

    Attributes:
        land_cover:  The predicted :class:`LandCoverClass` label.
        confidence:  Evidence-based confidence in ``[0.0, 1.0]``.
        evidence:    Human-readable strings describing the reasoning.
        features:    Computed spectral indices and statistics.
        warnings:    Non-fatal issues encountered during classification.
    """
    land_cover: str = LandCoverClass.UNKNOWN.value
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "land_cover": self.land_cover,
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "features": dict(self.features),
            "warnings": list(self.warnings),
        }


# ─── Band Mapping ───────────────────────────────────────────────────

# Common band-description tokens emitted by GDAL / rasterio drivers.
# Each key is a canonical SpectralBand; the value list is lowered tokens
# that may appear in raster band descriptions or colour interpretations.
_BAND_KEYWORDS: Dict[SpectralBand, List[str]] = {
    SpectralBand.RED:   ["red", "r", "band4", "b4", "band 4", "band_4", "red band"],
    SpectralBand.GREEN: ["green", "g", "band3", "b3", "band 3", "band_3", "green band"],
    SpectralBand.BLUE:  ["blue", "b", "band2", "b2", "band 2", "band_2", "blue band"],
    SpectralBand.NIR:   ["nir", "near infrared", "near-infrared", "band5", "b5",
                         "band 5", "band_5", "nir band", "nir1"],
    SpectralBand.SWIR:  ["swir", "shortwave infrared", "short-wave infrared",
                         "swir1", "swir2", "band6", "b6", "band 6", "band_6",
                         "band7", "b7", "band 7", "band_7"],
}

# Fallback positional assumption when metadata is absent.
# Indices are 0-based numpy axis-0 positions.
_DEFAULT_BAND_POSITIONS: Dict[SpectralBand, int] = {
    SpectralBand.RED:   0,
    SpectralBand.GREEN: 1,
    SpectralBand.BLUE:  2,
    SpectralBand.NIR:   3,
    SpectralBand.SWIR:  4,
}


class BandMapper:
    """Resolves canonical spectral band names to 0-based array indices.

    Resolution order:
        1. Explicit ``overrides`` passed by the caller.
        2. ``band_descriptions`` from raster metadata (fuzzy keyword match).
        3. ``color_interpretations`` from raster metadata.
        4. Positional fallback **only for RGB** (bands 0–2) when the raster
           has ≥3 bands. NIR / SWIR are *never* guessed positionally —
           ``None`` is returned instead so callers can handle the miss.

    Parameters:
        band_count:            Number of bands in the raster.
        band_descriptions:     Per-band description strings (may be empty/None).
        color_interpretations: Per-band colour interpretation strings (e.g.
                               ``["red", "green", "blue"]``).
        overrides:             Explicit mapping ``{SpectralBand: band_index}``.
    """

    def __init__(
        self,
        band_count: int,
        band_descriptions: Optional[List[Optional[str]]] = None,
        color_interpretations: Optional[List[Optional[str]]] = None,
        overrides: Optional[Dict[SpectralBand, int]] = None,
    ) -> None:
        self.band_count = band_count
        self._descriptions = band_descriptions or []
        self._color_interps = color_interpretations or []
        self._overrides = overrides or {}
        self._resolved: Dict[SpectralBand, Optional[int]] = {}
        self._resolve()

    # ── internal resolution ──────────────────────────────────────

    def _resolve(self) -> None:
        """Build the resolved map once at construction time."""
        used_indices: set[int] = set()

        # 1. Explicit overrides take top priority.
        for band, idx in self._overrides.items():
            if 0 <= idx < self.band_count:
                self._resolved[band] = idx
                used_indices.add(idx)

        # 2. Try band descriptions (fuzzy keyword matching).
        for band in SpectralBand:
            if band in self._resolved:
                continue
            matched_idx = self._match_keywords(band, self._descriptions)
            if matched_idx is not None and matched_idx not in used_indices:
                self._resolved[band] = matched_idx
                used_indices.add(matched_idx)

        # 3. Try colour interpretations.
        for band in SpectralBand:
            if band in self._resolved:
                continue
            matched_idx = self._match_keywords(band, self._color_interps)
            if matched_idx is not None and matched_idx not in used_indices:
                self._resolved[band] = matched_idx
                used_indices.add(matched_idx)

        # 4. Positional fallback — RGB only when ≥3 bands.
        rgb_bands = [SpectralBand.RED, SpectralBand.GREEN, SpectralBand.BLUE]
        if self.band_count >= 3:
            for band in rgb_bands:
                if band not in self._resolved:
                    pos = _DEFAULT_BAND_POSITIONS[band]
                    if pos not in used_indices:
                        self._resolved[band] = pos
                        used_indices.add(pos)

        # NIR / SWIR are intentionally NOT guessed positionally.

    def _match_keywords(
        self,
        band: SpectralBand,
        labels: List[Optional[str]],
    ) -> Optional[int]:
        """Return the 0-based index whose label matches ``band`` keywords."""
        keywords = _BAND_KEYWORDS.get(band, [])
        for idx, label in enumerate(labels):
            if label is None or idx >= self.band_count:
                continue
            normed = label.strip().lower()
            if normed in keywords:
                return idx
        return None

    # ── public API ───────────────────────────────────────────────

    def get(self, band: SpectralBand) -> Optional[int]:
        """Return the resolved 0-based index for *band*, or ``None``."""
        return self._resolved.get(band)

    def has(self, band: SpectralBand) -> bool:
        """Check whether *band* could be resolved."""
        return band in self._resolved

    def available_bands(self) -> List[SpectralBand]:
        """Return the list of successfully resolved bands."""
        return [b for b in SpectralBand if b in self._resolved]

    def missing_bands(self) -> List[SpectralBand]:
        """Return the list of bands that could NOT be resolved."""
        return [b for b in SpectralBand if b not in self._resolved]

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary of the mapping."""
        return {
            "band_count": self.band_count,
            "resolved": {b.value: self._resolved[b] for b in self._resolved},
            "missing": [b.value for b in self.missing_bands()],
        }


# ─── Statistics Helper ──────────────────────────────────────────────


def compute_regional_stats(pixels: np.ndarray) -> RegionalStats:
    """Compute robust regional statistics on a flat or 2-D pixel array.

    Parameters:
        pixels: Array of numeric pixel values.  Flattened internally.

    Returns:
        A :class:`RegionalStats` dataclass.
    """
    flat = pixels.ravel().astype(np.float64)
    if flat.size == 0:
        return RegionalStats()
    return RegionalStats(
        mean=float(np.mean(flat)),
        median=float(np.median(flat)),
        std=float(np.std(flat)),
        min=float(np.min(flat)),
        max=float(np.max(flat)),
        p10=float(np.percentile(flat, 10)),
        p25=float(np.percentile(flat, 25)),
        p75=float(np.percentile(flat, 75)),
        p90=float(np.percentile(flat, 90)),
    )


def _stats_to_dict(stats: RegionalStats) -> Dict[str, float]:
    """Convert a RegionalStats to a plain dict."""
    return {
        "mean": round(stats.mean, 6),
        "median": round(stats.median, 6),
        "std": round(stats.std, 6),
        "min": round(stats.min, 6),
        "max": round(stats.max, 6),
        "p10": round(stats.p10, 6),
        "p25": round(stats.p25, 6),
        "p75": round(stats.p75, 6),
        "p90": round(stats.p90, 6),
    }


# ─── Spectral Index Helpers ─────────────────────────────────────────

# Epsilon to prevent division-by-zero in normalised difference indices.
_EPS = 1e-10


def _normalised_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute (a − b) / (a + b + ε) element-wise."""
    return (a - b) / (a + b + _EPS)


# ─── Abstract Classifier ────────────────────────────────────────────


class BaseLandCoverClassifier(ABC):
    """Pluggable interface for land-cover classification.

    Concrete implementations (spectral baselines, ResNets, Transformers,
    foundation models) must implement :meth:`classify_pixels`.
    """

    @abstractmethod
    def classify_pixels(
        self,
        pixels: np.ndarray,
        band_descriptions: Optional[List[Optional[str]]] = None,
        color_interpretations: Optional[List[Optional[str]]] = None,
        band_overrides: Optional[Dict[SpectralBand, int]] = None,
    ) -> ClassificationResult:
        """Classify a pixel array and return a structured result.

        Parameters:
            pixels:               3-D NumPy array of shape
                                  ``(bands, height, width)``.
            band_descriptions:    Per-band description strings from raster
                                  metadata. May contain ``None`` entries.
            color_interpretations: Per-band colour interpretation labels.
            band_overrides:       Explicit ``{SpectralBand: index}`` map
                                  that takes precedence over metadata.

        Returns:
            A :class:`ClassificationResult`.
        """
        ...

    @property
    @abstractmethod
    def classifier_name(self) -> str:
        """Human-readable name for this classifier variant."""
        ...

    @property
    @abstractmethod
    def classifier_version(self) -> str:
        """Semver-style version string."""
        ...

    @property
    def is_trained_model(self) -> bool:  # noqa: D401
        """Whether this classifier is backed by a learned/trained model."""
        return False


# ─── Spectral Baseline Classifier ───────────────────────────────────


class SpectralLandCoverClassifier(BaseLandCoverClassifier):
    """Deterministic spectral-index classifier.

    Uses NDVI, NDWI, and NDBI to assign a land-cover label with
    evidence-based confidence.  Never fabricates spectral values for
    missing bands — missing indices degrade confidence and produce
    ``UNKNOWN`` when too few features are available.

    Spectral Indices
    ----------------
    NDVI = (NIR − RED) / (NIR + RED)
        Vegetation vigour indicator.
    NDWI = (GREEN − NIR) / (GREEN + NIR)
        Water body indicator.
    NDBI = (SWIR − NIR) / (SWIR + NIR)
        Built-up area indicator.
    """

    # ── Threshold Configuration ──────────────────────────────────
    # These are intentionally exposed as class attributes so that
    # subclasses or runtime patches can adjust them without forking.

    NDVI_VEGETATION_THRESHOLD: float = 0.25
    NDVI_AGRICULTURE_LOW: float = 0.15
    NDVI_AGRICULTURE_HIGH: float = 0.25
    NDWI_WATER_THRESHOLD: float = 0.10
    NDBI_BUILT_UP_THRESHOLD: float = 0.05
    BARE_SOIL_NDVI_CEILING: float = 0.10
    BARE_SOIL_NDWI_CEILING: float = 0.0
    ROAD_NDBI_THRESHOLD: float = 0.15
    ROAD_NDVI_CEILING: float = 0.05

    # ── Classifier Identity ──────────────────────────────────────

    @property
    def classifier_name(self) -> str:  # noqa: D401
        return "SpectralLandCoverClassifier"

    @property
    def classifier_version(self) -> str:  # noqa: D401
        return "1.0.0"

    @property
    def is_trained_model(self) -> bool:  # noqa: D401
        return False

    # ── Main Classification ──────────────────────────────────────

    def classify_pixels(
        self,
        pixels: np.ndarray,
        band_descriptions: Optional[List[Optional[str]]] = None,
        color_interpretations: Optional[List[Optional[str]]] = None,
        band_overrides: Optional[Dict[SpectralBand, int]] = None,
    ) -> ClassificationResult:
        """Classify a raster pixel region.

        Parameters:
            pixels:               ``(bands, height, width)`` float or int array.
            band_descriptions:    Metadata descriptions per band.
            color_interpretations: Colour interpretation labels per band.
            band_overrides:       Explicit band → index overrides.

        Returns:
            A :class:`ClassificationResult` with evidence-based confidence.
        """
        warnings: List[str] = []
        evidence: List[str] = []
        features: Dict[str, Any] = {}

        # ── Validate input shape ─────────────────────────────────
        if pixels.ndim != 3:
            return ClassificationResult(
                land_cover=LandCoverClass.UNKNOWN.value,
                confidence=0.0,
                evidence=["Input pixel array must be 3-D (bands, height, width)."],
                warnings=["Invalid input shape."],
            )

        n_bands, height, width = pixels.shape
        if height == 0 or width == 0:
            return ClassificationResult(
                land_cover=LandCoverClass.UNKNOWN.value,
                confidence=0.0,
                evidence=["Empty spatial extent."],
                warnings=["Pixel region has zero spatial extent."],
            )

        # ── Convert to float for safe arithmetic ─────────────────
        data = pixels.astype(np.float64)

        # ── Resolve band mapping ─────────────────────────────────
        mapper = BandMapper(
            band_count=n_bands,
            band_descriptions=band_descriptions,
            color_interpretations=color_interpretations,
            overrides=band_overrides,
        )
        features["band_mapping"] = mapper.summary()

        # ── Extract available bands ──────────────────────────────
        band_data: Dict[SpectralBand, np.ndarray] = {}
        for sb in SpectralBand:
            idx = mapper.get(sb)
            if idx is not None:
                band_data[sb] = data[idx]

        # ── Per-band regional statistics ─────────────────────────
        band_stats: Dict[str, Dict[str, float]] = {}
        for sb, arr in band_data.items():
            stats = compute_regional_stats(arr)
            band_stats[sb.value] = _stats_to_dict(stats)
        features["band_stats"] = band_stats

        # ── Compute spectral indices ─────────────────────────────
        ndvi: Optional[np.ndarray] = None
        ndwi: Optional[np.ndarray] = None
        ndbi: Optional[np.ndarray] = None

        has_nir = SpectralBand.NIR in band_data
        has_red = SpectralBand.RED in band_data
        has_green = SpectralBand.GREEN in band_data
        has_swir = SpectralBand.SWIR in band_data

        # NDVI
        if has_nir and has_red:
            ndvi = _normalised_difference(
                band_data[SpectralBand.NIR],
                band_data[SpectralBand.RED],
            )
            ndvi_stats = compute_regional_stats(ndvi)
            features["ndvi"] = _stats_to_dict(ndvi_stats)
            evidence.append(
                f"NDVI computed (mean={ndvi_stats.mean:.4f}, "
                f"median={ndvi_stats.median:.4f})."
            )
        else:
            missing = []
            if not has_nir:
                missing.append("NIR")
            if not has_red:
                missing.append("RED")
            warnings.append(
                f"Cannot compute NDVI: missing band(s) {', '.join(missing)}."
            )

        # NDWI
        if has_green and has_nir:
            ndwi = _normalised_difference(
                band_data[SpectralBand.GREEN],
                band_data[SpectralBand.NIR],
            )
            ndwi_stats = compute_regional_stats(ndwi)
            features["ndwi"] = _stats_to_dict(ndwi_stats)
            evidence.append(
                f"NDWI computed (mean={ndwi_stats.mean:.4f}, "
                f"median={ndwi_stats.median:.4f})."
            )
        else:
            missing = []
            if not has_green:
                missing.append("GREEN")
            if not has_nir:
                missing.append("NIR")
            warnings.append(
                f"Cannot compute NDWI: missing band(s) {', '.join(missing)}."
            )

        # NDBI
        if has_swir and has_nir:
            ndbi = _normalised_difference(
                band_data[SpectralBand.SWIR],
                band_data[SpectralBand.NIR],
            )
            ndbi_stats = compute_regional_stats(ndbi)
            features["ndbi"] = _stats_to_dict(ndbi_stats)
            evidence.append(
                f"NDBI computed (mean={ndbi_stats.mean:.4f}, "
                f"median={ndbi_stats.median:.4f})."
            )
        else:
            missing = []
            if not has_swir:
                missing.append("SWIR")
            if not has_nir:
                missing.append("NIR")
            warnings.append(
                f"Cannot compute NDBI: missing band(s) {', '.join(missing)}."
            )

        # ── Classify ─────────────────────────────────────────────
        res_tuple = self._apply_rules(
            ndvi=ndvi,
            ndwi=ndwi,
            ndbi=ndbi,
            band_data=band_data,
        )
        if len(res_tuple) == 4:
            label, conf, rule_evidence, ratios_dict = res_tuple
            features.update(ratios_dict)
        else:
            label, conf, rule_evidence = res_tuple

        evidence.extend(rule_evidence)
        features["available_indices"] = [
            name for name, arr in [("ndvi", ndvi), ("ndwi", ndwi), ("ndbi", ndbi)]
            if arr is not None
        ]

        return ClassificationResult(
            land_cover=label.value,
            confidence=round(conf, 4),
            evidence=evidence,
            features=features,
            warnings=warnings,
        )

    # ── Rule Engine ──────────────────────────────────────────────

    def _apply_rules(
        self,
        ndvi: Optional[np.ndarray],
        ndwi: Optional[np.ndarray],
        ndbi: Optional[np.ndarray],
        band_data: Dict[SpectralBand, np.ndarray],
    ) -> Tuple[LandCoverClass, float, List[str]]:
        """Apply threshold rules and return (label, confidence, evidence).

        Confidence is computed from the *margin* between the index value
        and the threshold — larger margins yield higher confidence.
        When fewer indices are available the maximum confidence is capped.
        """
        evidence: List[str] = []
        candidates: List[Tuple[LandCoverClass, float, str]] = []

        n_available = sum(1 for x in (ndvi, ndwi, ndbi) if x is not None)

        # Confidence ceiling based on available information.
        if n_available >= 3:
            conf_cap = 1.0
        elif n_available == 2:
            conf_cap = 0.75
        elif n_available == 1:
            conf_cap = 0.55
        else:
            # RGB Color-Space Heuristic Fallback for 3-Band Optical Imagery
            has_r = SpectralBand.RED in band_data
            has_g = SpectralBand.GREEN in band_data
            has_b = SpectralBand.BLUE in band_data
            
            if has_r and has_g and has_b:
                r_arr = band_data[SpectralBand.RED]
                g_arr = band_data[SpectralBand.GREEN]
                b_arr = band_data[SpectralBand.BLUE]

                r_mean = float(np.mean(r_arr))
                g_mean = float(np.mean(g_arr))
                b_mean = float(np.mean(b_arr))
                
                # Spatial pixel-level masks for 3-band RGB optical imagery
                # 1. Vegetation mask: Green dominance (2G - R - B > 5 and G > B + 2)
                is_veg = (2.0 * g_arr - r_arr - b_arr > 5.0) | (g_arr > b_arr + 2)
                vvi_mask = (2.0 * g_arr - r_arr - b_arr > 10.0) & (g_arr > r_arr)
                veg_ratio = float(np.mean(vvi_mask))

                # 2. Strict Water mask (excluding vegetation & tree canopy shadows)
                water_pixels = (~is_veg) & (b_arr > r_arr + 15) & (b_arr > g_arr + 5) & (b_arr > 65)
                deep_water = (~is_veg) & (r_arr < 40) & (g_arr < 50) & (b_arr > r_arr + 15) & (b_arr > 45)
                water_mask = water_pixels | deep_water
                raw_water = float(np.mean(water_mask))
                water_ratio = raw_water if raw_water >= 0.04 else 0.0

                # 3. Built-up / Structure mask: High brightness or neutral grey/white/red rooftop roofs
                brightness = (r_arr + g_arr + b_arr) / 3.0
                rgb_std = np.std([r_arr, g_arr, b_arr], axis=0)
                built_mask = ((brightness > 95.0) & (rgb_std < 25.0)) | ((r_arr > g_arr + 15.0) & (r_arr > b_arr + 15.0) & (r_arr > 110.0))
                built_ratio = float(np.mean(built_mask))

                # 4. Bare soil / Agricultural plot mask
                soil_mask = (r_arr > g_arr) & (r_arr > b_arr) & (brightness < 120.0) & (~vvi_mask) & (~water_mask)
                soil_ratio = float(np.mean(soil_mask))

                ratios_dict = {
                    "water_ratio": water_ratio,
                    "veg_ratio": veg_ratio,
                    "built_ratio": built_ratio,
                    "soil_ratio": soil_ratio
                }
                evidence.append(f"RGB Spatial Ratios: Water={water_ratio*100:.1f}%, Vegetation={veg_ratio*100:.1f}%, Built-Up={built_ratio*100:.1f}%, Soil={soil_ratio*100:.1f}%.")

                if water_ratio > 0.12 and water_ratio >= veg_ratio and water_ratio >= built_ratio:
                    return LandCoverClass.WATER, min(0.70 + water_ratio * 0.25, 0.92), evidence, ratios_dict
                elif veg_ratio > 0.20 and veg_ratio >= built_ratio:
                    return LandCoverClass.VEGETATION, min(0.70 + veg_ratio * 0.25, 0.90), evidence, ratios_dict
                elif built_ratio > 0.15:
                    return LandCoverClass.BUILT_UP, min(0.70 + built_ratio * 0.25, 0.90), evidence, ratios_dict
                elif soil_ratio > 0.25:
                    return LandCoverClass.BARE_SOIL, min(0.65 + soil_ratio * 0.25, 0.85), evidence, ratios_dict
                else:
                    # Default dominant ratio
                    ratios = [("WATER", water_ratio), ("VEGETATION", veg_ratio), ("BUILT_UP", built_ratio), ("BARE_SOIL", soil_ratio)]
                    ratios.sort(key=lambda x: x[1], reverse=True)
                    top_class = ratios[0][0]
                    return LandCoverClass(top_class), 0.75, evidence, ratios_dict
            else:
                evidence.append("No spectral indices or RGB channels could be computed; returning UNKNOWN.")
                return LandCoverClass.UNKNOWN, 0.0, evidence

        # ── Water ────────────────────────────────────────────────
        if ndwi is not None:
            mean_ndwi = float(np.mean(ndwi))
            if mean_ndwi > self.NDWI_WATER_THRESHOLD:
                ndvi_ok = True
                if ndvi is not None:
                    mean_ndvi = float(np.mean(ndvi))
                    ndvi_ok = mean_ndvi < self.NDVI_VEGETATION_THRESHOLD
                if ndvi_ok:
                    margin = (mean_ndwi - self.NDWI_WATER_THRESHOLD) / (
                        1.0 - self.NDWI_WATER_THRESHOLD + _EPS
                    )
                    conf = min(0.60 + 0.40 * margin, conf_cap)
                    candidates.append((
                        LandCoverClass.WATER,
                        conf,
                        f"NDWI mean ({mean_ndwi:.4f}) > threshold "
                        f"({self.NDWI_WATER_THRESHOLD}), suggesting water.",
                    ))

        # ── Vegetation ───────────────────────────────────────────
        if ndvi is not None:
            mean_ndvi = float(np.mean(ndvi))
            if mean_ndvi >= self.NDVI_VEGETATION_THRESHOLD:
                margin = (mean_ndvi - self.NDVI_VEGETATION_THRESHOLD) / (
                    1.0 - self.NDVI_VEGETATION_THRESHOLD + _EPS
                )
                conf = min(0.55 + 0.40 * margin, conf_cap)
                candidates.append((
                    LandCoverClass.VEGETATION,
                    conf,
                    f"NDVI mean ({mean_ndvi:.4f}) ≥ threshold "
                    f"({self.NDVI_VEGETATION_THRESHOLD}), indicating "
                    f"strong vegetation.",
                ))

        # ── Agriculture ──────────────────────────────────────────
        if ndvi is not None:
            mean_ndvi = float(np.mean(ndvi))
            std_ndvi = float(np.std(ndvi))
            if (
                self.NDVI_AGRICULTURE_LOW <= mean_ndvi < self.NDVI_AGRICULTURE_HIGH
                and std_ndvi < 0.15
            ):
                uniformity_bonus = max(0.0, 0.10 - std_ndvi) * 2
                conf = min(0.50 + uniformity_bonus, conf_cap)
                candidates.append((
                    LandCoverClass.AGRICULTURE,
                    conf,
                    f"NDVI mean ({mean_ndvi:.4f}) in agricultural range "
                    f"[{self.NDVI_AGRICULTURE_LOW}, {self.NDVI_AGRICULTURE_HIGH}) "
                    f"with low std ({std_ndvi:.4f}), suggesting managed crops.",
                ))

        # ── Built-up / Road Infrastructure ───────────────────────
        if ndbi is not None:
            mean_ndbi = float(np.mean(ndbi))
            if mean_ndbi > self.NDBI_BUILT_UP_THRESHOLD:
                margin = (mean_ndbi - self.NDBI_BUILT_UP_THRESHOLD) / (
                    1.0 - self.NDBI_BUILT_UP_THRESHOLD + _EPS
                )

                # Distinguish road from general built-up.
                is_road = False
                if ndvi is not None:
                    mean_ndvi_here = float(np.mean(ndvi))
                    if (
                        mean_ndbi > self.ROAD_NDBI_THRESHOLD
                        and mean_ndvi_here < self.ROAD_NDVI_CEILING
                    ):
                        is_road = True

                if is_road:
                    conf = min(0.55 + 0.35 * margin, conf_cap)
                    candidates.append((
                        LandCoverClass.ROAD_INFRASTRUCTURE,
                        conf,
                        f"NDBI mean ({mean_ndbi:.4f}) > road threshold "
                        f"({self.ROAD_NDBI_THRESHOLD}) with very low NDVI, "
                        f"suggesting road/infrastructure.",
                    ))
                else:
                    conf = min(0.55 + 0.35 * margin, conf_cap)
                    candidates.append((
                        LandCoverClass.BUILT_UP,
                        conf,
                        f"NDBI mean ({mean_ndbi:.4f}) > threshold "
                        f"({self.NDBI_BUILT_UP_THRESHOLD}), suggesting "
                        f"built-up area.",
                    ))

        # ── Bare Soil ────────────────────────────────────────────
        if ndvi is not None:
            mean_ndvi = float(np.mean(ndvi))
            ndwi_check_pass = True
            if ndwi is not None:
                mean_ndwi = float(np.mean(ndwi))
                ndwi_check_pass = mean_ndwi < self.BARE_SOIL_NDWI_CEILING
            ndbi_check_pass = True
            if ndbi is not None:
                mean_ndbi = float(np.mean(ndbi))
                ndbi_check_pass = mean_ndbi <= self.NDBI_BUILT_UP_THRESHOLD

            if (
                mean_ndvi < self.BARE_SOIL_NDVI_CEILING
                and ndwi_check_pass
                and ndbi_check_pass
            ):
                conf = min(0.50, conf_cap)
                candidates.append((
                    LandCoverClass.BARE_SOIL,
                    conf,
                    f"NDVI mean ({mean_ndvi:.4f}) < bare-soil ceiling "
                    f"({self.BARE_SOIL_NDVI_CEILING}), low NDWI and NDBI, "
                    f"suggesting bare soil.",
                ))

        # ── Pick the winner ──────────────────────────────────────
        if not candidates:
            evidence.append(
                "No spectral rule matched decisively; classified as UNKNOWN."
            )
            return LandCoverClass.UNKNOWN, min(0.10, conf_cap), evidence

        # Sort by confidence descending; first match wins.
        candidates.sort(key=lambda c: c[1], reverse=True)
        winner_label, winner_conf, winner_reason = candidates[0]
        evidence.append(winner_reason)

        if len(candidates) > 1:
            runner = candidates[1]
            evidence.append(
                f"Runner-up candidate: {runner[0].value} "
                f"(confidence={runner[1]:.4f}) — {runner[2]}"
            )

        return winner_label, winner_conf, evidence
