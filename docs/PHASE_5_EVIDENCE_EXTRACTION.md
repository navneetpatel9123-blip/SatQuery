# Phase 5: Evidence Extraction Engine

## Overview
The Evidence Extraction Engine acts as the "audit" layer of the SatQuery AI bi-temporal change analysis pipeline. 
While Phase 3 determines **where** change occurred, and Phase 4 determines **what** changed, Phase 5 determines **why** the pipeline believes this classification.

It acts strictly as an auditable extractor—reusing actual spectral and statistical calculations generated during Phase 4—to prevent hallucinations or fabricated "AI logic."

## Architecture

The engine is completely decoupled from FastAPI endpoints, accepting native NumPy arrays and Phase 3/4 DataClasses. 

```python
extract(t1_pixels, t2_pixels, change_regions, phase4_results, t1_date, t2_date) -> List[EvidenceResult]
```

### Integration Flow (Phase 3 → Phase 4 → Phase 5)
1. **Phase 3:** Outputs bounding box `ChangeRegion` masks of detected anomalies.
2. **Phase 4:** The `ChangePipelineService` parses the rasters and executes the `SpectralLandCoverClassifier` and `RuleBasedChangeUnderstander` to determine land-cover transitions and change types.
3. **Phase 5:** The pipeline executes the `RuleBasedEvidenceExtractor`, providing the arrays and the Phase 4 results. The extractor compiles the structured evidence and attaches it back to each region.

### API Endpoint
```http
POST /api/v1/change/evidence
```
**Input:** `{ "analysis_id": "...", "t1_image_id": "...", "t2_image_id": "..." }`
**Action:** Executes the end-to-end Phase 4 + Phase 5 pipeline for a stored Phase 3 `analysis_id`.
**Output:** An aggregated response containing execution traces, Phase 4 classifications, and deeply structured Phase 5 `evidence` attributes embedded in each region.

### Execution Trace
The pipeline appends the following operational events directly to the result's trace log:
`PHASE_4_COMPLETED` → `PHASE_5_STARTED` → `EVIDENCE_EXTRACTION_STARTED` → `EVIDENCE_RESULTS_ATTACHED` → `PHASE_5_COMPLETED`.

### Components
1. **BaseEvidenceExtractor**: Abstract base class.
2. **RuleBasedEvidenceExtractor**: Primary implementation containing deterministic rule-based checks mapping Phase 4 heuristics to human-readable structured evidence objects.

## Evidence Categories
The engine isolates evidence into distinct categories for rendering in future frontend widgets:

- **SPATIAL**: Extract bounds, geospatial area (sq meters), pixel area, centroids, and proportional change statistics.
- **SPECTRAL**: Consume Phase 4 indices (NDVI, NDWI, NDBI). Calculate deltas and provide deterministic interpretation ("vegetation indicator decreased").
- **STATISTICAL**: Provide standard band-wise statistics (Mean, StdDev, Min, Max) comparing T1 vs T2 distributions.
- **LAND_COVER**: Format direct Phase 4 semantic transitions and their associated raw confidences.
- **TEMPORAL**: Parse ISO-8601 acquisition timestamps to report chronological elapsed time.
- **VISUAL**: Emit precise bounding box arrays (c_start, r_start, c_end, r_end) to allow down-stream clients or the web-dashboard to generate local cropped previews.

## Evidence Scoring & Contradiction Detection

The engine calculates an independent `evidence_score` separate from the raw classifier confidence.

**Detection Rules (Example: Built-Up Expansion):**
- Checks if `NDBI` has increased > 0.05 (Supports).
- Checks if `NDBI` has strongly decreased < -0.05 (Contradicts).

If a contradiction occurs, the evidence score is penalized (-0.2 per contradiction) and a pipeline-level warning is appended: *"Spectral evidence partially contradicts the proposed change classification."*

## Output Formats
The `EvidenceResult` includes an `evidence_summary` formatted cleanly for database persistence, containing:
- `overall_strength`: [HIGH, MEDIUM, LOW, INSUFFICIENT]
- `supporting_evidence`: List of verified string statements.
- `contradicting_evidence`: List of detected contradictions.
- `explanation`: Deterministic human-readable paragraph generated natively from structured data (NOT an LLM).

## Limitations
- **Baseline Nature**: This implementation is a rules-based deterministic engine, not a learned evidential machine-learning model.
- **Resolution Coupling**: Geospatial area falls back to pixel area if CRS/resolution information fails parsing during ingestion.
- **Missing Bands**: For non-multispectral/SAR datasets missing NIR/SWIR, optical indices are gracefully marked `available=False` and do not contribute to contradiction penalties.
