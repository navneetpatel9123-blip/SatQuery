"""
Unit tests for the Phase 4 / Task 2 — Change Understander.

Uses small synthetic NumPy arrays as raster fixtures.  No disk I/O,
no FastAPI dependency, no large dataset downloads.
"""

import pytest
import numpy as np
from typing import Dict, List, Optional

from app.models.land_cover_classifier import (
    ClassificationResult,
    LandCoverClass,
    SpectralBand,
    SpectralLandCoverClassifier,
)
from app.models.change_understander import (
    BaseChangeUnderstander,
    ChangeConfidence,
    ChangeRegionInput,
    ChangeTypeClass,
    ChangeUnderstandingResult,
    ConfidenceLabel,
    FeatureEvidence,
    RegionChangeResult,
    RuleBasedChangeUnderstander,
    TraceEvent,
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
) -> np.ndarray:
    """Build a synthetic ``(bands, height, width)`` pixel array."""
    bands = [
        np.full((height, width), red, dtype=np.float64),
        np.full((height, width), green, dtype=np.float64),
        np.full((height, width), blue, dtype=np.float64),
    ]
    if nir is not None:
        bands.append(np.full((height, width), nir, dtype=np.float64))
    if swir is not None:
        bands.append(np.full((height, width), swir, dtype=np.float64))
    return np.stack(bands, axis=0)


# Standard 5-band descriptions.
_FULL = ["red", "green", "blue", "nir", "swir"]
_4BAND = ["red", "green", "blue", "nir"]


def _region(
    region_id: str = "R001",
    bbox: Optional[List[float]] = None,
    area: float = 100.0,
    p3_conf: float = 0.80,
) -> ChangeRegionInput:
    """Build a ``ChangeRegionInput`` with sensible defaults.

    The default ``bbox`` covers the full image as normalised coordinates.
    """
    return ChangeRegionInput(
        region_id=region_id,
        bbox=bbox or [0.0, 0.0, 1.0, 1.0],
        centroid=[0.5, 0.5],
        area_sq_m=area,
        phase3_confidence=p3_conf,
        phase3_change_type="other",
        phase3_description="Test region.",
    )


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def understander() -> RuleBasedChangeUnderstander:
    return RuleBasedChangeUnderstander()


# Canonical spectral signatures (R, G, B, NIR, SWIR).
_WATER   = dict(red=30,  green=120, blue=80,  nir=20,  swir=15)
_VEG     = dict(red=30,  green=60,  blue=40,  nir=200, swir=50)
_AGRI    = dict(red=50,  green=55,  blue=45,  nir=70,  swir=40)
_BUILT   = dict(red=100, green=90,  blue=85,  nir=120, swir=150)
_BARE    = dict(red=100, green=95,  blue=90,  nir=105, swir=100)
_ROAD    = dict(red=120, green=110, blue=100, nir=130, swir=200)


# ─── 1. Agriculture → Built-up ──────────────────────────────────────


class TestAgriToBuiltUp:
    def test_change_type(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        r = result.regions[0]
        assert r.t1_land_cover == LandCoverClass.AGRICULTURE.value
        assert r.t2_land_cover == LandCoverClass.BUILT_UP.value
        assert r.change_type == ChangeTypeClass.BUILT_UP_EXPANSION.value

    def test_evidence_mentions_expansion(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert any("expansion" in e.lower() for e in result.regions[0].evidence)


# ─── 2. Vegetation → Built-up ───────────────────────────────────────


class TestVegToBuiltUp:
    def test_change_type(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.BUILT_UP_EXPANSION.value


# ─── 3. Vegetation → Bare Soil ──────────────────────────────────────


class TestVegToBareSoil:
    def test_deforestation(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.DEFORESTATION.value

    def test_ndvi_decreased(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        ndvi_delta = [
            fd for fd in result.regions[0].feature_deltas
            if fd.feature == "NDVI"
        ]
        assert len(ndvi_delta) == 1
        assert ndvi_delta[0].delta < 0  # NDVI should decrease.


# ─── 4. Built-up → Bare Soil ────────────────────────────────────────


class TestBuiltUpToBareSoil:
    def test_demolition(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_BUILT)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.DEMOLITION.value


# ─── 5. Water → Non-water ───────────────────────────────────────────


class TestWaterToNonWater:
    def test_water_body_change(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_WATER)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.WATER_BODY_CHANGE.value


# ─── 6. Non-water → Water ───────────────────────────────────────────


class TestNonWaterToWater:
    def test_water_body_change(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_WATER)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.WATER_BODY_CHANGE.value


# ─── 7. Vegetation Gain ─────────────────────────────────────────────


class TestVegetationGain:
    def test_bare_to_veg(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_BARE)
        t2 = _make_pixels(**_VEG)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.VEGETATION_GAIN.value

    def test_ndvi_increased(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_BARE)
        t2 = _make_pixels(**_VEG)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        ndvi_delta = [
            fd for fd in result.regions[0].feature_deltas
            if fd.feature == "NDVI"
        ]
        assert len(ndvi_delta) == 1
        assert ndvi_delta[0].delta > 0


# ─── 8. Agriculture → Bare Soil ─────────────────────────────────────


class TestAgriToBareSoil:
    def test_agricultural_conversion(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.AGRICULTURAL_CONVERSION.value


# ─── 9. Unknown / Ambiguous Transition ──────────────────────────────


class TestUnknownTransition:
    def test_same_class_is_unknown_change(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_VEG)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert result.regions[0].change_type == ChangeTypeClass.UNKNOWN_CHANGE.value

    def test_rgb_only_low_info(self, understander: RuleBasedChangeUnderstander):
        """RGB-only → many indices unavailable → expect warnings."""
        t1 = _make_pixels(red=60, green=80, blue=70)
        t2 = _make_pixels(red=120, green=100, blue=90)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=["red", "green", "blue"]
        )
        # Should have warnings about missing bands.
        r = result.regions[0]
        assert any("NIR" in w for w in r.warnings)

    def test_single_band_sar_guard(
        self, understander: RuleBasedChangeUnderstander
    ):
        """Single-band imagery should produce a top-level SAR warning."""
        t1 = np.full((1, 8, 8), 100.0)
        t2 = np.full((1, 8, 8), 200.0)
        result = understander.understand_changes(t1, t2, [_region()])
        assert any("SAR" in w or "single" in w.lower() for w in result.warnings)


# ─── 10. Missing NIR ────────────────────────────────────────────────


class TestMissingNIR:
    def test_warnings_propagated(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(red=30, green=120, blue=80)
        t2 = _make_pixels(red=100, green=90, blue=85)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=["red", "green", "blue"]
        )
        r = result.regions[0]
        assert any("NIR" in w for w in r.warnings)

    def test_no_ndvi_fabricated(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(red=30, green=120, blue=80)
        t2 = _make_pixels(red=100, green=90, blue=85)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=["red", "green", "blue"]
        )
        r = result.regions[0]
        # No NDVI delta should exist.
        assert not any(fd.feature == "NDVI" for fd in r.feature_deltas)


# ─── 11. Missing SWIR ───────────────────────────────────────────────


class TestMissingSWIR:
    def test_swir_warning(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(red=30, green=60, blue=40, nir=200)
        t2 = _make_pixels(red=100, green=95, blue=90, nir=105)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_4BAND
        )
        r = result.regions[0]
        assert any("SWIR" in w for w in r.warnings)

    def test_no_ndbi_delta(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(red=30, green=60, blue=40, nir=200)
        t2 = _make_pixels(red=100, green=95, blue=90, nir=105)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_4BAND
        )
        r = result.regions[0]
        assert not any(fd.feature == "NDBI" for fd in r.feature_deltas)


# ─── 12. Multiple Regions ───────────────────────────────────────────


class TestMultipleRegions:
    def test_processes_all_regions(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG, height=16, width=16)
        t2 = _make_pixels(**_BUILT, height=16, width=16)
        regions = [
            _region("R001", bbox=[0.0, 0.0, 0.5, 0.5]),
            _region("R002", bbox=[0.5, 0.5, 1.0, 1.0]),
            _region("R003", bbox=[0.0, 0.5, 0.5, 1.0]),
        ]
        result = understander.understand_changes(
            t1, t2, regions, band_descriptions=_FULL
        )
        assert len(result.regions) == 3
        assert result.summary["total_regions"] == 3
        ids = {r.region_id for r in result.regions}
        assert ids == {"R001", "R002", "R003"}

    def test_summary_change_counts(
        self, understander: RuleBasedChangeUnderstander
    ):
        # Two regions: veg→built, water→bare
        t1_a = _make_pixels(**_VEG, height=16, width=16)
        t2_a = _make_pixels(**_BUILT, height=16, width=16)
        regions = [
            _region("R001", bbox=[0.0, 0.0, 0.5, 0.5]),
            _region("R002", bbox=[0.5, 0.5, 1.0, 1.0]),
        ]
        result = understander.understand_changes(
            t1_a, t2_a, regions, band_descriptions=_FULL
        )
        counts = result.summary["change_type_counts"]
        assert isinstance(counts, dict)
        # All regions get the same change type (uniform pixels).
        total = sum(counts.values())
        assert total == 2


# ─── 13. Confidence Calculation ──────────────────────────────────────


class TestConfidenceCalculation:
    def test_confidence_bounded(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        conf = result.regions[0].change_confidence
        assert 0.0 <= conf.score <= 1.0

    def test_confidence_has_factors(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        conf = result.regions[0].change_confidence
        assert len(conf.factors) >= 3
        assert conf.label in [cl.value for cl in ConfidenceLabel]

    def test_deterministic(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BUILT)
        r1 = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        r2 = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        assert r1.regions[0].change_confidence.score == r2.regions[0].change_confidence.score

    def test_high_confidence_for_clear_transition(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region(p3_conf=0.90)], band_descriptions=_FULL
        )
        conf = result.regions[0].change_confidence
        assert conf.label in [ConfidenceLabel.HIGH.value, ConfidenceLabel.MEDIUM.value]


# ─── 14. Evidence Generation ────────────────────────────────────────


class TestEvidenceGeneration:
    def test_evidence_list_populated(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        r = result.regions[0]
        assert len(r.evidence) > 0

    def test_feature_deltas_have_structure(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        r = result.regions[0]
        for fd in r.feature_deltas:
            assert isinstance(fd.feature, str)
            assert isinstance(fd.t1, float)
            assert isinstance(fd.t2, float)
            assert isinstance(fd.delta, float)
            assert isinstance(fd.interpretation, str)

    def test_feature_evidence_to_dict(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BARE)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        r = result.regions[0]
        for fd in r.feature_deltas:
            d = fd.to_dict()
            assert set(d.keys()) == {
                "feature", "t1", "t2", "delta", "interpretation"
            }


# ─── 15. Description Generation ─────────────────────────────────────


class TestDescriptionGeneration:
    def test_description_nonempty(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region("R042")], band_descriptions=_FULL
        )
        desc = result.regions[0].description
        assert len(desc) > 0
        assert "R042" in desc

    def test_description_mentions_classes(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        desc = result.regions[0].description.lower()
        assert "vegetation" in desc
        assert "built" in desc

    def test_no_change_description(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_VEG)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        desc = result.regions[0].description.lower()
        assert "no land-cover transition" in desc


# ─── 16. Phase 3 Output Compatibility ───────────────────────────────


class TestPhase3Compatibility:
    def test_adapter_from_change_region(self):
        """ChangeRegionInput.from_phase3_change_region works with
        the Phase 3 ``ChangeRegion`` Pydantic model."""
        from app.schemas import ChangeRegion, ChangeType

        cr = ChangeRegion(
            region_id="P3-R001",
            change_type=ChangeType.BUILT_UP_EXPANSION,
            area_sq_m=250.5,
            centroid=[-122.45, 37.75],
            bbox=[-122.5, 37.7, -122.4, 37.8],
            confidence=0.85,
            description="Phase 3 detected built-up expansion.",
        )
        adapted = ChangeRegionInput.from_phase3_change_region(cr)
        assert adapted.region_id == "P3-R001"
        assert adapted.bbox == [-122.5, 37.7, -122.4, 37.8]
        assert adapted.centroid == [-122.45, 37.75]
        assert adapted.area_sq_m == 250.5
        assert adapted.phase3_confidence == 0.85
        assert adapted.phase3_change_type == "built_up_expansion"

    def test_adapter_works_with_plain_object(self):
        """Adapter accepts any duck-typed object with the right attrs."""

        class FakeRegion:
            region_id = "FAKE-001"
            change_type = "other"
            area_sq_m = 50.0
            centroid = [0.0, 0.0]
            bbox = [0, 0, 10, 10]
            confidence = 0.5
            description = "fake"

        adapted = ChangeRegionInput.from_phase3_change_region(FakeRegion())
        assert adapted.region_id == "FAKE-001"

    def test_result_to_dict_shape(
        self, understander: RuleBasedChangeUnderstander
    ):
        """End-to-end result serialises cleanly."""
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        d = result.to_dict()
        assert "analysis_id" in d
        assert "regions" in d
        assert "summary" in d
        assert "execution_trace" in d
        assert "warnings" in d
        region = d["regions"][0]
        expected_keys = {
            "region_id", "bbox", "centroid", "pixel_area",
            "geospatial_area", "t1_land_cover", "t1_land_cover_confidence",
            "t2_land_cover", "t2_land_cover_confidence", "change_type",
            "change_confidence", "spectral_features_t1",
            "spectral_features_t2", "feature_deltas", "evidence",
            "description", "warnings",
        }
        assert expected_keys.issubset(set(region.keys()))


# ─── 17. Execution Trace ────────────────────────────────────────────


class TestExecutionTrace:
    def test_trace_events_present(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        events = [t.event for t in result.execution_trace]
        assert "CHANGE_REGIONS_RECEIVED" in events
        assert "REGION_EXTRACTED" in events
        assert "T1_CLASSIFICATION" in events
        assert "T2_CLASSIFICATION" in events
        assert "FEATURE_COMPARISON" in events
        assert "CHANGE_RULE_EVALUATION" in events
        assert "CHANGE_CLASSIFICATION" in events
        assert "CONFIDENCE_CALCULATION" in events
        assert "DESCRIPTION_GENERATED" in events
        assert "RESULT_COMPLETED" in events

    def test_trace_to_dict(self, understander: RuleBasedChangeUnderstander):
        t1 = _make_pixels(**_AGRI)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [_region()], band_descriptions=_FULL
        )
        for te in result.execution_trace:
            d = te.to_dict()
            assert "event" in d
            assert "timestamp" in d
            assert "duration_ms" in d


# ─── 18. Pluggable Interface ────────────────────────────────────────


class TestPluggableInterface:
    def test_is_subclass(self):
        assert issubclass(RuleBasedChangeUnderstander, BaseChangeUnderstander)

    def test_not_trained_model(
        self, understander: RuleBasedChangeUnderstander
    ):
        assert understander.is_trained_model is False

    def test_identity(self, understander: RuleBasedChangeUnderstander):
        assert understander.understander_name == "RuleBasedChangeUnderstander"
        assert understander.understander_version == "1.0.0"


# ─── 19. Edge Cases ─────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_regions_list(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = _make_pixels(**_VEG)
        t2 = _make_pixels(**_BUILT)
        result = understander.understand_changes(
            t1, t2, [], band_descriptions=_FULL
        )
        assert len(result.regions) == 0
        assert result.summary["total_regions"] == 0

    def test_invalid_input_shape(
        self, understander: RuleBasedChangeUnderstander
    ):
        t1 = np.zeros((8, 8), dtype=np.float64)  # 2-D
        t2 = np.zeros((8, 8), dtype=np.float64)
        result = understander.understand_changes(t1, t2, [_region()])
        assert len(result.warnings) > 0
