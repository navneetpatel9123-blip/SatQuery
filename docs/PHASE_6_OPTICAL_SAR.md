# Phase 6 Task 2: Optical-SAR Cross-Modal Analysis

**Status**: 🟢 COMPLETED (130/130 Tests Passing)

## Overview
This document outlines the architecture and baseline implementation for fusing co-registered Optical/Multispectral and SAR imagery to provide enriched, multi-modal evidence for land cover classification (e.g., built-up, water).

> [!IMPORTANT]
> This is a **rule-based baseline method** utilizing deterministic thresholds and spectral indices. It is **NOT** a learned ML model. A future `LearnedFusionModel` may replace this baseline using the exact same pluggable `BaseOpticalSARAnalyzer` interface.

## Architecture

### `RuleBasedOpticalSARAnalyzer`
1. **Co-registration Validation**: Checks CRS equality, bounds overlap, dimensionality, and spatial transform matrices. 
   - Strict identical grids yield `CO_REGISTERED`.
   - Different but overlapping grids yield `ALIGNMENT_REQUIRED`.
   - Distinct, non-overlapping bounds yield `INCOMPATIBLE` which automatically halts the pipeline.
2. **Optical Feature Extraction**: Integrates seamlessly with Phase 4's `SpectralLandCoverClassifier` to safely extract NDVI, NDWI, NDBI, and dominant land-cover evidence.
3. **SAR Feature Extraction**: Analyzes spatial backscatter distributions by returning mean, median, min, max, std, and detected polarizations (e.g., VV, VH).
4. **Cross-Modal Fusion Engine**: Generates a composite classification taking both modalities into account.

### Agreement Classifications
To support evidence-based explainability, every fused region returns an `AgreementScore`:
- `STRONG_AGREEMENT`: Both sensors confirm the classification (e.g., high NDBI from Optical + high mean backscatter from SAR).
- `CONFLICT`: Sensors strongly disagree (e.g., high NDWI from Optical indicating water + high backscatter from SAR contradicting water).
- `OPTICAL_DOMINANT` / `SAR_DOMINANT`: Situations where one sensor provides the vast majority of the discriminative evidence.

## Orchestrator & Registry Updates
The `ModelRegistry` now tracks the `optical_sar_analyzer` capability. The `PipelineOrchestrator` validates exactly two images (one optical, one SAR) and executes the sequence with a transparent execution trace:
1. `OPTICAL_SAR_STARTED`
2. `OPTICAL_INPUT_VALIDATED`
3. `SAR_INPUT_VALIDATED`
4. `COREGISTRATION_CHECKED`
5. `OPTICAL_FEATURES_EXTRACTED`
6. `SAR_FEATURES_EXTRACTED`
7. `CROSS_MODAL_FUSION_STARTED`
8. `CROSS_MODAL_EVIDENCE_COMPUTED`
9. `AGREEMENT_ANALYZED`
10. `OPTICAL_SAR_COMPLETED`

## Limitations & Future Work
- **Thresholds**: Currently, the thresholds for determining "high" and "low" SAR backscatter are hardcoded normalized defaults (0.5 for high, 0.1 for low) intended for MVP testing. Production usage requires calibration against true decibel (dB) thresholds based on sensor calibration.
- **Resampling**: When grids are `ALIGNMENT_REQUIRED`, the baseline does not yet perform sub-pixel nearest-neighbor resampling.
- **Future Upgrade**: A trained multi-modal transformer/ResNet should be integrated via a subclass of `BaseOpticalSARAnalyzer` to learn complex textural fusions rather than relying on strict scalar rules.
