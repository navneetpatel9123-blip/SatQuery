import numpy as np
import pytest

from app.models.change_understander import ChangeRegionInput, RegionChangeResult, ChangeConfidence
from app.models.evidence_extractor import RuleBasedEvidenceExtractor


@pytest.fixture
def sample_arrays():
    t1 = np.random.rand(5, 64, 64) * 255
    t2 = np.random.rand(5, 64, 64) * 255
    return t1, t2


@pytest.fixture
def sample_regions():
    return [
        ChangeRegionInput(
            region_id="R001",
            bbox=[0.1, 0.1, 0.5, 0.5],
            centroid=[0.3, 0.3],
            area_sq_m=1500.0,
            phase3_confidence=0.9
        ),
        ChangeRegionInput(
            region_id="R002",
            bbox=[0.6, 0.6, 0.9, 0.9],
            centroid=[0.75, 0.75],
            area_sq_m=0.0, # Test fallback area
            phase3_confidence=0.8
        )
    ]


@pytest.fixture
def sample_phase4_results():
    return [
        RegionChangeResult(
            region_id="R001",
            bbox=[0.1, 0.1, 0.5, 0.5],
            pixel_area=400,
            t1_land_cover="AGRICULTURE",
            t2_land_cover="BUILT_UP",
            change_type="BUILT_UP_EXPANSION",
            change_confidence=ChangeConfidence(score=0.85),
            spectral_features_t1={"NDVI": 0.5, "NDWI": -0.2, "NDBI": -0.1},
            spectral_features_t2={"NDVI": 0.1, "NDWI": -0.3, "NDBI": 0.3}
        ),
        RegionChangeResult(
            region_id="R002",
            bbox=[0.6, 0.6, 0.9, 0.9],
            pixel_area=300,
            t1_land_cover="VEGETATION",
            t2_land_cover="BARE_SOIL",
            change_type="DEFORESTATION",
            change_confidence=ChangeConfidence(score=0.9),
            spectral_features_t1={"NDVI": 0.7, "NDWI": 0.0, "NDBI": -0.2},
            spectral_features_t2={"NDVI": 0.1, "NDWI": 0.0, "NDBI": 0.1}
        )
    ]


def test_spatial_evidence(sample_arrays, sample_regions, sample_phase4_results):
    extractor = RuleBasedEvidenceExtractor()
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results)
    
    assert len(results) == 2
    r1 = results[0].spatial_evidence
    assert r1.geospatial_area == 1500.0
    assert r1.percent_total_changed_area == 100.0 # R002 has area 0
    assert "width" in r1.dimensions
    
    r2 = results[1].spatial_evidence
    assert r2.geospatial_area is None # Fallback


def test_spectral_evidence(sample_arrays, sample_regions, sample_phase4_results):
    extractor = RuleBasedEvidenceExtractor()
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results)
    
    r1_spectral = results[0].spectral_evidence
    ndbi = next(s for s in r1_spectral if s.feature == "NDBI")
    assert ndbi.delta > 0
    assert "increased" in ndbi.interpretation
    
    ndvi = next(s for s in r1_spectral if s.feature == "NDVI")
    assert ndvi.delta < 0
    assert "decreased" in ndvi.interpretation


def test_statistical_evidence(sample_arrays, sample_regions, sample_phase4_results):
    extractor = RuleBasedEvidenceExtractor()
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results)
    
    stats = results[0].statistical_evidence
    assert len(stats) == 5 # 5 bands
    assert hasattr(stats[0], 't1_mean')
    assert hasattr(stats[0], 'delta_mean')


def test_land_cover_evidence(sample_arrays, sample_regions, sample_phase4_results):
    extractor = RuleBasedEvidenceExtractor()
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results)
    
    lc = results[0].land_cover_evidence
    assert lc.t1_class == "AGRICULTURE"
    assert lc.t2_class == "BUILT_UP"
    assert "AGRICULTURE to BUILT_UP" in lc.transition_description


def test_contradictory_evidence(sample_arrays, sample_regions):
    extractor = RuleBasedEvidenceExtractor()
    
    # Force contradiction: Built-up expansion but NDBI strongly decreases
    p4_results = [
        RegionChangeResult(
            region_id="R001",
            bbox=[0.1, 0.1, 0.5, 0.5],
            change_type="BUILT_UP_EXPANSION",
            change_confidence=ChangeConfidence(score=0.85),
            spectral_features_t1={"NDBI": 0.5},
            spectral_features_t2={"NDBI": 0.1}
        )
    ]
    
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions[:1], p4_results)
    assert len(results[0].contradicting_evidence) > 0
    assert "NDBI decreased" in results[0].contradicting_evidence[0]
    assert any("contradicts" in w for w in results[0].warnings)


def test_missing_bands(sample_arrays, sample_regions):
    extractor = RuleBasedEvidenceExtractor()
    
    p4_results = [
        RegionChangeResult(
            region_id="R001",
            bbox=[0.1, 0.1, 0.5, 0.5],
            change_type="BUILT_UP_EXPANSION",
            # Empty spectral features = missing bands
            spectral_features_t1={},
            spectral_features_t2={}
        )
    ]
    
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions[:1], p4_results)
    ndwi = next(s for s in results[0].spectral_evidence if s.feature == "NDWI")
    assert not ndwi.available
    assert "unavailable" in ndwi.reason


def test_temporal_evidence(sample_arrays, sample_regions, sample_phase4_results):
    extractor = RuleBasedEvidenceExtractor()
    
    # With dates
    results = extractor.extract(
        sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results,
        t1_date="2023-01-01T00:00:00", t2_date="2023-02-01T00:00:00"
    )
    assert results[0].temporal_evidence.days_elapsed == 31
    
    # Missing dates
    results_missing = extractor.extract(
        sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results,
    )
    assert results_missing[0].temporal_evidence.days_elapsed is None
    assert "unavailable" in results_missing[0].temporal_evidence.description


def test_execution_trace(sample_arrays, sample_regions, sample_phase4_results):
    extractor = RuleBasedEvidenceExtractor()
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results)
    
    trace_events = [t.event for t in results[0].execution_trace]
    assert "EVIDENCE_EXTRACTION_STARTED" in trace_events
    assert "SPATIAL_EVIDENCE_EXTRACTED" in trace_events
    assert "EVIDENCE_SCORE_CALCULATED" in trace_events
    assert "EVIDENCE_EXTRACTION_COMPLETED" in trace_events


def test_visual_crop_references(sample_arrays, sample_regions, sample_phase4_results):
    extractor = RuleBasedEvidenceExtractor()
    results = extractor.extract(sample_arrays[0], sample_arrays[1], sample_regions, sample_phase4_results)
    
    visual = results[0].visual_evidence
    assert visual.bounding_box == [0.1, 0.1, 0.5, 0.5]
    # W=64, H=64. Bbox [0.1, 0.1, 0.5, 0.5] -> c_start=6, r_start=6, c_end=32, r_end=32
    assert visual.t1_crop_window == [6, 6, 32, 32]
