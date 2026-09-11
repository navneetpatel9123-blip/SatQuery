# Phase 4: Change Understanding

This document describes the design, implementation, and limitations of the
Change Understanding subsystem (Phase 4 / Task 2) for the SatQuery AI platform.

> **Important**: The current implementation is a **transparent spectral/rule-based
> baseline** and **NOT** a trained ML change-understanding model. All confidence
> scores are evidence-based, not calibrated probabilities.

---

## 1. Architecture

```
              Phase 3 Change Regions
                       |
                       v
            ChangeRegionInput Adapter
                       |
                       v
        BaseChangeUnderstander (ABC)
                       |
                       v
        RuleBasedChangeUnderstander
           /           |          \
          v            v           v
    T1 extraction  T2 extraction  Band mapping
          |            |
          v            v
    SpectralLandCover  SpectralLandCover
    Classifier (T1)    Classifier (T2)
          |            |
          +-----+------+
                |
                v
        Feature Delta Computation
                |
                v
        Transition Rule Engine
                |
                v
        Confidence Calculation
                |
                v
        Description Generation
                |
                v
        ChangeUnderstandingResult
```

### Key Design Decisions

- **Reuses Task 1 classifier**: The `SpectralLandCoverClassifier` from
  `backend/app/models/land_cover_classifier.py` is used directly — no
  duplication of spectral classification logic.
- **FastAPI-independent**: All classes are plain Python; no web framework
  coupling.
- **Phase 3 adapter**: `ChangeRegionInput.from_phase3_change_region()` converts
  the Phase 3 `ChangeRegion` Pydantic model into the internal format, keeping
  the understander decoupled.
- **Pluggable ABC**: Future implementations (ML, VLM, Hybrid) inherit from
  `BaseChangeUnderstander`.

---

## 2. Change Taxonomy

| Change Type                    | Example Transition             |
|--------------------------------|--------------------------------|
| `BUILT_UP_EXPANSION`          | Agriculture → Built-up         |
| `DEMOLITION`                  | Built-up → Bare Soil           |
| `VEGETATION_GAIN`             | Bare Soil → Vegetation         |
| `DEFORESTATION`               | Vegetation → Bare Soil         |
| `WATER_BODY_CHANGE`           | Any ↔ Water                    |
| `AGRICULTURAL_CONVERSION`     | Agriculture → Bare Soil        |
| `ROAD_INFRASTRUCTURE_DEVELOPMENT` | Agriculture → Road         |
| `UNKNOWN_CHANGE`              | Ambiguous / same class         |

---

## 3. Per-Region Classification Flow

For each Phase 3 change region, the pipeline executes:

1. **Extract T1 region** — pixel window from the pre-change raster.
2. **Extract T2 region** — pixel window from the post-change raster.
3. **Classify T1** — via `SpectralLandCoverClassifier.classify_pixels()`.
4. **Classify T2** — same classifier on the T2 window.
5. **Compute feature deltas** — NDVI, NDWI, NDBI mean differences.
6. **Apply transition rules** — deterministic table lookup.
7. **Calculate confidence** — weighted combination of 5 factors.
8. **Generate description** — templated natural-language string.
9. **Assemble result** — `RegionChangeResult` dataclass.

---

## 4. Evidence Model

Each region returns:

```json
{
  "feature_deltas": [
    {
      "feature": "NDVI",
      "t1": 0.73,
      "t2": 0.04,
      "delta": -0.69,
      "interpretation": "NDVI decreased"
    }
  ],
  "evidence": [
    "T1 classified as VEGETATION (confidence=0.85).",
    "T2 classified as BARE_SOIL (confidence=0.50).",
    "NDVI: NDVI decreased (Δ=-0.6900).",
    "Rule: Vegetated area transitioned to bare soil..."
  ]
}
```

Evidence is structured and traceable. No LLM-generated text.

---

## 5. Confidence Model

Confidence is a **weighted evidence score** (NOT a calibrated probability).

| Factor                        | Weight | Source                              |
|-------------------------------|--------|-------------------------------------|
| T1 classification confidence  | 0.25   | SpectralLandCoverClassifier         |
| T2 classification confidence  | 0.25   | SpectralLandCoverClassifier         |
| Feature-change strength       | 0.20   | Max |Δ| across NDVI/NDWI/NDBI      |
| Phase 3 detection confidence  | 0.15   | ChangeRegionInput.phase3_confidence |
| Rule agreement                | 0.15   | 1.0 if rule matched, 0.1 otherwise |

Confidence tiers:

| Label  | Score Range  |
|--------|-------------|
| HIGH   | ≥ 0.70      |
| MEDIUM | 0.40 – 0.69 |
| LOW    | < 0.40      |

---

## 6. Limitations

1. **Rule-based only** — no learned features or deep representations.
   Accuracy is bounded by spectral-index thresholds.
2. **No sub-pixel analysis** — mixed land-cover pixels are classified by
   dominant spectral signal.
3. **No texture or spatial context** — classification is purely spectral;
   spatial patterns (e.g., building footprints, road linearity) are not used.
4. **RGB-only imagery degrades significantly** — NDVI, NDWI, and NDBI all
   require NIR; NDBI also requires SWIR. Missing bands reduce the confidence
   cap and may result in `UNKNOWN_CHANGE`.
5. **SAR not supported** — single-band SAR data bypasses optical indices
   entirely and produces warnings.
6. **No temporal decay modelling** — the system does not account for seasonal
   variation or intermediate states between T1 and T2.
7. **Transition rules are exhaustive but simplified** — some real-world
   transitions (e.g., partial urbanisation within agricultural land) may
   not match cleanly.

---

## 7. Future ML Integration

The `BaseChangeUnderstander` ABC is designed for drop-in replacements:

```python
class MLChangeUnderstander(BaseChangeUnderstander):
    """Trained bi-temporal change understanding model."""

    def understand_changes(self, t1_pixels, t2_pixels, change_regions, **kw):
        # Load trained model weights
        # Run inference on T1/T2 pairs
        # Return ChangeUnderstandingResult
        ...
```

Planned future implementations:

| Variant                  | Description                                     |
|--------------------------|-------------------------------------------------|
| `MLChangeUnderstander`   | Supervised CNN/Transformer trained on change data|
| `VLMChangeUnderstander`  | Vision-Language Model for open-vocabulary change |
| `HybridChangeUnderstander` | Ensemble of rule-based + ML approaches        |

---

## 8. Files

| File                                      | Purpose                              |
|-------------------------------------------|---------------------------------------|
| `backend/app/models/change_understander.py` | Core implementation                 |
| `backend/app/models/land_cover_classifier.py` | Spectral classifier (Task 1)      |
| `backend/tests/test_change_understander.py` | 40 unit tests                       |
| `backend/tests/test_land_cover_classifier.py` | 34 unit tests (Task 1)            |

---

## 9. Verification

```bash
# Run Phase 4 Task 2 tests only
python -m pytest backend/tests/test_change_understander.py -v

# Run all project tests
python -m pytest backend/tests/ -v
```
