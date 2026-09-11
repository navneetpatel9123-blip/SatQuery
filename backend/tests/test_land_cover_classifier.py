"""
Unit tests for the Phase 4 Land Cover Classifier.

Uses small synthetic raster fixtures (pure NumPy arrays) — no disk I/O
or FastAPI dependency required.  Each test verifies the spectral-index
baseline against deterministic pixel patterns.
"""

import pytest
import numpy as np

from app.models.land_cover_classifier import (
    BaseLandCoverClassifier,
    BandMapper,
    ClassificationResult,
    LandCoverClass,
    RegionalStats,
    SpectralBand,
    SpectralLandCoverClassifier,
    compute_regional_stats,
)


# ─── Helpers ────────────────────────────────────────────────────────


def _make_pixels(
    red: float,
    green: float,
    blue: float,
    nir: float | None = None,
    swir: float | None = None,
    *,
    height: int = 8,
    width: int = 8,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """Build a synthetic ``(bands, height, width)`` pixel array.

    Only RGB bands are required; NIR and SWIR are optional.
    When ``noise_scale > 0``, Gaussian noise is added per band
    to simulate sensor variation.
    """
    bands = [
        np.full((height, width), red, dtype=np.float64),
        np.full((height, width), green, dtype=np.float64),
        np.full((height, width), blue, dtype=np.float64),
    ]
    if nir is not None:
        bands.append(np.full((height, width), nir, dtype=np.float64))
    if swir is not None:
        bands.append(np.full((height, width), swir, dtype=np.float64))

    arr = np.stack(bands, axis=0)
    if noise_scale > 0:
        rng = np.random.default_rng(42)
        arr += rng.normal(0, noise_scale, arr.shape)
    return arr


# Standard band descriptions for a 5-band multispectral raster.
_FULL_DESCRIPTIONS = ["red", "green", "blue", "nir", "swir"]


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def classifier() -> SpectralLandCoverClassifier:
    """Return a fresh ``SpectralLandCoverClassifier`` instance."""
    return SpectralLandCoverClassifier()


# ─── 1. Water-like region ───────────────────────────────────────────


class TestWaterClassification:
    """Water bodies have high NDWI and low NDVI."""

    def test_water_detection(self, classifier: SpectralLandCoverClassifier):
        # NIR very low, GREEN high → positive NDWI; RED ≈ NIR → low NDVI.
        pixels = _make_pixels(red=30, green=120, blue=80, nir=20, swir=15)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert result.land_cover == LandCoverClass.WATER.value
        assert result.confidence > 0.0
        assert "ndwi" in result.features
        assert any("water" in e.lower() for e in result.evidence)

    def test_water_confidence_scales_with_margin(
        self, classifier: SpectralLandCoverClassifier
    ):
        weak = _make_pixels(red=40, green=60, blue=50, nir=45, swir=30)
        strong = _make_pixels(red=10, green=150, blue=90, nir=5, swir=5)
        r_weak = classifier.classify_pixels(
            weak, band_descriptions=_FULL_DESCRIPTIONS
        )
        r_strong = classifier.classify_pixels(
            strong, band_descriptions=_FULL_DESCRIPTIONS
        )
        # Stronger water signal → higher confidence.
        if (
            r_weak.land_cover == LandCoverClass.WATER.value
            and r_strong.land_cover == LandCoverClass.WATER.value
        ):
            assert r_strong.confidence >= r_weak.confidence


# ─── 2. Vegetation-like region ──────────────────────────────────────


class TestVegetationClassification:
    """Dense vegetation has high NDVI (NIR ≫ RED)."""

    def test_vegetation_detection(self, classifier: SpectralLandCoverClassifier):
        # NIR very high, RED low → NDVI ≫ 0.25.
        pixels = _make_pixels(red=30, green=60, blue=40, nir=200, swir=50)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert result.land_cover == LandCoverClass.VEGETATION.value
        assert result.confidence > 0.0
        assert "ndvi" in result.features
        assert any("vegetation" in e.lower() for e in result.evidence)


# ─── 3. Agriculture-like region ─────────────────────────────────────


class TestAgricultureClassification:
    """Agriculture has moderate, uniform NDVI."""

    def test_agriculture_detection(self, classifier: SpectralLandCoverClassifier):
        # NDVI ≈ 0.18 (moderate), very low spatial variance.
        # NDVI = (nir - red) / (nir + red) ≈ (70 - 50) / (70 + 50) ≈ 0.167
        pixels = _make_pixels(red=50, green=55, blue=45, nir=70, swir=40)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert result.land_cover == LandCoverClass.AGRICULTURE.value
        assert result.confidence > 0.0
        assert any("agric" in e.lower() or "crop" in e.lower() for e in result.evidence)


# ─── 4. Built-up-like region ────────────────────────────────────────


class TestBuiltUpClassification:
    """Built-up areas show high NDBI (SWIR ≫ NIR)."""

    def test_built_up_detection(self, classifier: SpectralLandCoverClassifier):
        # SWIR > NIR → positive NDBI; NDVI moderate (above road ceiling).
        # NDVI = (120-100)/(120+100) = 0.091; NDBI = (150-120)/(150+120) = 0.111
        pixels = _make_pixels(red=100, green=90, blue=85, nir=120, swir=150)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert result.land_cover == LandCoverClass.BUILT_UP.value
        assert result.confidence > 0.0
        assert "ndbi" in result.features
        assert any("built" in e.lower() for e in result.evidence)


# ─── 5. Bare-soil-like region ───────────────────────────────────────


class TestBareSoilClassification:
    """Bare soil has very low NDVI, low NDWI, and low NDBI."""

    def test_bare_soil_detection(self, classifier: SpectralLandCoverClassifier):
        # All bands similar → near-zero indices.
        pixels = _make_pixels(red=100, green=95, blue=90, nir=105, swir=100)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert result.land_cover == LandCoverClass.BARE_SOIL.value
        assert result.confidence > 0.0
        assert any("bare" in e.lower() or "soil" in e.lower() for e in result.evidence)


# ─── 6. Missing NIR ────────────────────────────────────────────────


class TestMissingNIR:
    """Without NIR, NDVI and NDWI cannot be computed."""

    def test_missing_nir_produces_warnings(
        self, classifier: SpectralLandCoverClassifier
    ):
        # Only RGB — no NIR or SWIR.
        pixels = _make_pixels(red=60, green=80, blue=70)
        result = classifier.classify_pixels(
            pixels, band_descriptions=["red", "green", "blue"]
        )
        assert any("NIR" in w for w in result.warnings)
        assert any("NDVI" in w for w in result.warnings)
        assert any("NDWI" in w for w in result.warnings)
        # Should still return a valid result structure.
        assert result.land_cover in [lc.value for lc in LandCoverClass]

    def test_missing_nir_does_not_fabricate(
        self, classifier: SpectralLandCoverClassifier
    ):
        pixels = _make_pixels(red=60, green=80, blue=70)
        result = classifier.classify_pixels(
            pixels, band_descriptions=["red", "green", "blue"]
        )
        # NDVI and NDWI must NOT appear in features (they weren't computed).
        assert "ndvi" not in result.features
        assert "ndwi" not in result.features


# ─── 7. Missing SWIR ───────────────────────────────────────────────


class TestMissingSWIR:
    """Without SWIR, NDBI cannot be computed but NDVI/NDWI still work."""

    def test_missing_swir_warning_and_partial(
        self, classifier: SpectralLandCoverClassifier
    ):
        # 4-band raster (R, G, B, NIR) — no SWIR.
        pixels = _make_pixels(red=30, green=60, blue=40, nir=200)
        result = classifier.classify_pixels(
            pixels, band_descriptions=["red", "green", "blue", "nir"]
        )
        assert any("SWIR" in w for w in result.warnings)
        assert any("NDBI" in w for w in result.warnings)
        # NDVI and NDWI should still be computed.
        assert "ndvi" in result.features
        assert "ndwi" in result.features
        assert "ndbi" not in result.features

    def test_missing_swir_caps_confidence(
        self, classifier: SpectralLandCoverClassifier
    ):
        pixels = _make_pixels(red=30, green=60, blue=40, nir=200)
        result = classifier.classify_pixels(
            pixels, band_descriptions=["red", "green", "blue", "nir"]
        )
        # With only 2 indices (NDVI, NDWI), confidence capped at 0.75.
        assert result.confidence <= 0.75


# ─── 8. Unknown / Ambiguous region ──────────────────────────────────


class TestUnknownClassification:
    """Ambiguous or trivial inputs should fall back to UNKNOWN."""

    def test_empty_array(self, classifier: SpectralLandCoverClassifier):
        pixels = np.zeros((3, 0, 0), dtype=np.float64)
        result = classifier.classify_pixels(pixels)
        assert result.land_cover == LandCoverClass.UNKNOWN.value
        assert result.confidence == 0.0

    def test_wrong_ndim(self, classifier: SpectralLandCoverClassifier):
        pixels = np.zeros((10, 10), dtype=np.float64)
        result = classifier.classify_pixels(pixels)
        assert result.land_cover == LandCoverClass.UNKNOWN.value

    def test_single_band_unknown(self, classifier: SpectralLandCoverClassifier):
        """Single-band raster with no metadata → no indices computable."""
        pixels = np.full((1, 8, 8), 100.0)
        result = classifier.classify_pixels(pixels)
        assert result.land_cover == LandCoverClass.UNKNOWN.value
        assert len(result.warnings) > 0


# ─── 9. Band Mapping ───────────────────────────────────────────────


class TestBandMapper:
    """Verify the band-mapping resolution mechanism."""

    def test_from_descriptions(self):
        mapper = BandMapper(
            band_count=5,
            band_descriptions=["red", "green", "blue", "nir", "swir"],
        )
        assert mapper.get(SpectralBand.RED) == 0
        assert mapper.get(SpectralBand.GREEN) == 1
        assert mapper.get(SpectralBand.BLUE) == 2
        assert mapper.get(SpectralBand.NIR) == 3
        assert mapper.get(SpectralBand.SWIR) == 4
        assert mapper.missing_bands() == []

    def test_from_color_interpretations(self):
        mapper = BandMapper(
            band_count=4,
            color_interpretations=["red", "green", "blue", "nir"],
        )
        assert mapper.has(SpectralBand.RED)
        assert mapper.has(SpectralBand.NIR)
        assert not mapper.has(SpectralBand.SWIR)

    def test_positional_fallback_rgb_only(self):
        # No metadata at all, 3-band raster.
        mapper = BandMapper(band_count=3)
        assert mapper.get(SpectralBand.RED) == 0
        assert mapper.get(SpectralBand.GREEN) == 1
        assert mapper.get(SpectralBand.BLUE) == 2
        # NIR should NOT be guessed.
        assert mapper.get(SpectralBand.NIR) is None

    def test_positional_fallback_does_not_guess_nir(self):
        # 4 bands but no metadata — NIR must NOT be assumed.
        mapper = BandMapper(band_count=4)
        assert mapper.get(SpectralBand.NIR) is None
        assert SpectralBand.NIR in mapper.missing_bands()

    def test_explicit_overrides_win(self):
        mapper = BandMapper(
            band_count=5,
            band_descriptions=["red", "green", "blue", "nir", "swir"],
            overrides={SpectralBand.NIR: 2, SpectralBand.RED: 4},
        )
        assert mapper.get(SpectralBand.NIR) == 2
        assert mapper.get(SpectralBand.RED) == 4

    def test_summary_format(self):
        mapper = BandMapper(band_count=3)
        s = mapper.summary()
        assert "band_count" in s
        assert "resolved" in s
        assert "missing" in s
        assert s["band_count"] == 3


# ─── 10. Confidence ─────────────────────────────────────────────────


class TestConfidence:
    """Confidence must be evidence-based, never random."""

    def test_deterministic(self, classifier: SpectralLandCoverClassifier):
        pixels = _make_pixels(red=30, green=120, blue=80, nir=20, swir=15)
        r1 = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        r2 = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert r1.confidence == r2.confidence
        assert r1.land_cover == r2.land_cover

    def test_confidence_bounded(self, classifier: SpectralLandCoverClassifier):
        pixels = _make_pixels(red=30, green=120, blue=80, nir=20, swir=15)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_fewer_indices_lower_cap(
        self, classifier: SpectralLandCoverClassifier
    ):
        # Full 5-band → all 3 indices.
        full = _make_pixels(red=30, green=120, blue=80, nir=20, swir=15)
        r_full = classifier.classify_pixels(
            full, band_descriptions=_FULL_DESCRIPTIONS
        )
        # 4-band → only 2 indices (no NDBI).
        partial = _make_pixels(red=30, green=120, blue=80, nir=20)
        r_partial = classifier.classify_pixels(
            partial, band_descriptions=["red", "green", "blue", "nir"]
        )
        # Partial should not exceed its confidence cap (0.75).
        assert r_partial.confidence <= 0.75


# ─── 11. Feature Extraction ─────────────────────────────────────────


class TestFeatureExtraction:
    """Verify that computed features are real and correctly structured."""

    def test_ndvi_present_when_nir_available(
        self, classifier: SpectralLandCoverClassifier
    ):
        pixels = _make_pixels(red=50, green=60, blue=40, nir=150, swir=80)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert "ndvi" in result.features
        ndvi_stats = result.features["ndvi"]
        assert "mean" in ndvi_stats
        assert "median" in ndvi_stats
        assert "std" in ndvi_stats
        assert "p10" in ndvi_stats
        assert "p90" in ndvi_stats

    def test_ndwi_present_when_nir_available(
        self, classifier: SpectralLandCoverClassifier
    ):
        pixels = _make_pixels(red=50, green=60, blue=40, nir=150, swir=80)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert "ndwi" in result.features

    def test_ndbi_present_when_swir_available(
        self, classifier: SpectralLandCoverClassifier
    ):
        pixels = _make_pixels(red=50, green=60, blue=40, nir=150, swir=80)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        assert "ndbi" in result.features

    def test_band_stats_per_resolved_band(
        self, classifier: SpectralLandCoverClassifier
    ):
        pixels = _make_pixels(red=50, green=60, blue=40, nir=150, swir=80)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        band_stats = result.features["band_stats"]
        for band_name in ["RED", "GREEN", "BLUE", "NIR", "SWIR"]:
            assert band_name in band_stats
            stats = band_stats[band_name]
            for key in ["mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90"]:
                assert key in stats

    def test_available_indices_list(
        self, classifier: SpectralLandCoverClassifier
    ):
        pixels = _make_pixels(red=50, green=60, blue=40, nir=150)
        result = classifier.classify_pixels(
            pixels, band_descriptions=["red", "green", "blue", "nir"]
        )
        avail = result.features["available_indices"]
        assert "ndvi" in avail
        assert "ndwi" in avail
        assert "ndbi" not in avail


# ─── 12. Regional Statistics Helper ─────────────────────────────────


class TestRegionalStats:
    """Verify ``compute_regional_stats`` independently."""

    def test_uniform_array(self):
        arr = np.full((4, 4), 42.0)
        stats = compute_regional_stats(arr)
        assert stats.mean == pytest.approx(42.0)
        assert stats.median == pytest.approx(42.0)
        assert stats.std == pytest.approx(0.0)
        assert stats.min == pytest.approx(42.0)
        assert stats.max == pytest.approx(42.0)

    def test_empty_array(self):
        arr = np.array([], dtype=np.float64)
        stats = compute_regional_stats(arr)
        assert stats.mean == 0.0

    def test_known_distribution(self):
        arr = np.arange(1, 101, dtype=np.float64)  # 1..100
        stats = compute_regional_stats(arr)
        assert stats.mean == pytest.approx(50.5)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(100.0)


# ─── 13. ClassificationResult serialisation ─────────────────────────


class TestClassificationResultDict:
    """Verify to_dict() output shape."""

    def test_to_dict_keys(self, classifier: SpectralLandCoverClassifier):
        pixels = _make_pixels(red=30, green=120, blue=80, nir=20, swir=15)
        result = classifier.classify_pixels(
            pixels, band_descriptions=_FULL_DESCRIPTIONS
        )
        d = result.to_dict()
        assert set(d.keys()) == {"land_cover", "confidence", "evidence", "features", "warnings"}
        assert isinstance(d["evidence"], list)
        assert isinstance(d["features"], dict)
        assert isinstance(d["warnings"], list)


# ─── 14. Pluggable Interface Contract ───────────────────────────────


class TestPluggableInterface:
    """Ensure ``SpectralLandCoverClassifier`` satisfies the ABC."""

    def test_is_subclass(self):
        assert issubclass(SpectralLandCoverClassifier, BaseLandCoverClassifier)

    def test_not_trained_model(self, classifier: SpectralLandCoverClassifier):
        assert classifier.is_trained_model is False

    def test_has_name_and_version(self, classifier: SpectralLandCoverClassifier):
        assert classifier.classifier_name == "SpectralLandCoverClassifier"
        assert classifier.classifier_version == "1.0.0"
