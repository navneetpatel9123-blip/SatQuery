"""
SatQuery AI — Change Understanding Pipeline Service
====================================================

Integrates the Phase 3 change-detection output with the Phase 4
Land Cover Classifier and Change Understander into a single
end-to-end pipeline.

    Phase 3 Change Regions
            ↓
    T1/T2 raster loading
            ↓
    SpectralLandCoverClassifier (per region)
            ↓
    RuleBasedChangeUnderstander
            ↓
    Structured ChangeUnderstandingResult
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio

from app.config import settings
from app.models.land_cover_classifier import (
    SpectralBand,
    SpectralLandCoverClassifier,
)
from app.models.change_understander import (
    ChangeRegionInput,
    ChangeUnderstandingResult,
    RuleBasedChangeUnderstander,
    TraceEvent,
)
from app.models.evidence_extractor import (
    EvidenceResult,
    RuleBasedEvidenceExtractor,
)
from app.schemas import (
    AnalysisResult,
    ChangeRegion,
    RasterAsset,
    TraceNode,
)


# ─── Pipeline Result ────────────────────────────────────────────────


class ChangePipelineResult:
    """Wraps ``ChangeUnderstandingResult`` with pipeline-level metadata.

    Attributes:
        analysis_id:       Unique pipeline run identifier.
        phase:             Always ``"change_understanding"``.
        result:            The core ``ChangeUnderstandingResult``.
        models_used:       Registry metadata for the models involved.
        pipeline_trace:    Top-level pipeline execution trace nodes.
        warnings:          Pipeline-level warnings.
    """

    def __init__(
        self,
        analysis_id: str,
        result: ChangeUnderstandingResult,
        models_used: List[Dict[str, str]],
        pipeline_trace: List[Dict[str, Any]],
        warnings: List[str],
        evidence_results: Optional[List[EvidenceResult]] = None,
    ) -> None:
        self.analysis_id = analysis_id
        self.phase = "evidence_extraction"
        self.result = result
        self.models_used = models_used
        self.pipeline_trace = pipeline_trace
        self.warnings = warnings
        self.evidence_results = evidence_results or []

    def to_dict(self) -> Dict[str, Any]:
        inner = self.result.to_dict()
        
        # Attach evidence to regions
        regions_out = []
        evidence_map = {e.region_id: e for e in self.evidence_results}
        
        # Track summary stats for Phase 5
        high, med, low = 0, 0, 0
        all_trace = self.pipeline_trace.copy()
        all_trace.extend([t.to_dict() if hasattr(t, "to_dict") else t for t in self.result.execution_trace])
        
        for r in inner.get("regions", []):
            rid = r.get("region_id")
            ev = evidence_map.get(rid)
            if ev:
                r["evidence"] = ev.to_dict()
                label = ev.evidence_label
                if label == "HIGH": high += 1
                elif label == "MEDIUM": med += 1
                elif label == "LOW": low += 1
                
                # Append evidence trace
                all_trace.extend([t.to_dict() if hasattr(t, "to_dict") else t for t in ev.execution_trace])
                
            regions_out.append(r)
            
        summary = inner.get("summary", {})
        summary["high_evidence_regions"] = high
        summary["medium_evidence_regions"] = med
        summary["low_evidence_regions"] = low

        return {
            "analysis_id": self.analysis_id,
            "phase": self.phase,
            "regions": regions_out,
            "summary": summary,
            "models": self.models_used,
            "warnings": self.warnings + inner.get("warnings", []),
            "execution_trace": all_trace,
        }


# ─── Pipeline Service ───────────────────────────────────────────────


class ChangePipelineService:
    """Orchestrates the Phase 3 → Phase 4 change-understanding flow.

    Public API:
        ``run_pipeline(t1_asset, t2_asset, change_regions, ...)``
            — Execute the full pipeline on real raster data.

        ``run_pipeline_from_arrays(t1_pixels, t2_pixels, change_regions, ...)``
            — Execute on pre-loaded NumPy arrays (for testing / direct use).
    """

    def __init__(
        self,
        classifier: Optional[SpectralLandCoverClassifier] = None,
        understander: Optional[RuleBasedChangeUnderstander] = None,
        extractor: Optional[RuleBasedEvidenceExtractor] = None,
    ) -> None:
        self._classifier = classifier or SpectralLandCoverClassifier()
        self._understander = understander or RuleBasedChangeUnderstander(
            classifier=self._classifier
        )
        self._extractor = extractor or RuleBasedEvidenceExtractor()

    # ── High-Level: from RasterAssets ────────────────────────────

    def run_pipeline(
        self,
        t1_asset: RasterAsset,
        t2_asset: RasterAsset,
        change_regions: List[ChangeRegion],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> ChangePipelineResult:
        """Run the full Phase 3 → Phase 4 pipeline from raster assets.

        Parameters:
            t1_asset:       The T1 (pre-change) raster asset.
            t2_asset:       The T2 (post-change) raster asset.
            change_regions: Phase 3 ``ChangeRegion`` instances.
            options:        Optional configuration overrides.

        Returns:
            A :class:`ChangePipelineResult`.
        """
        analysis_id = str(uuid.uuid4())
        warnings: List[str] = []
        trace: List[Dict[str, Any]] = []

        # ── Validate inputs ──────────────────────────────────────
        trace.append(self._trace_dict(
            "PHASE_3_COMPLETED", "",
            {"change_regions_count": len(change_regions)},
        ))

        if not change_regions:
            warnings.append("No change regions provided by Phase 3.")
            trace.append(self._trace_dict(
                "PHASE_4_COMPLETED", "",
                {"status": "skipped", "reason": "no_regions"},
            ))
            return ChangePipelineResult(
                analysis_id=analysis_id,
                result=ChangeUnderstandingResult(
                    analysis_id=analysis_id,
                    summary={"total_regions": 0, "change_type_counts": {},
                             "total_changed_pixels": 0, "confidence_summary": {},
                             "warning_count": 1},
                    warnings=warnings,
                ),
                models_used=self._models_metadata(),
                pipeline_trace=trace,
                warnings=warnings,
            )

        trace.append(self._trace_dict(
            "CHANGE_REGIONS_AVAILABLE", "",
            {"region_ids": [cr.region_id for cr in change_regions]},
        ))

        # ── Load raster data ─────────────────────────────────────
        t0 = time.time_ns()
        t1_pixels, t1_descs, t1_interps, t1_warns = self._load_raster(t1_asset)
        warnings.extend(t1_warns)

        t2_pixels, t2_descs, t2_interps, t2_warns = self._load_raster(t2_asset)
        warnings.extend(t2_warns)
        load_ms = (time.time_ns() - t0) / 1e6

        if t1_pixels is None or t2_pixels is None:
            missing = []
            if t1_pixels is None:
                missing.append("T1")
            if t2_pixels is None:
                missing.append("T2")
            warnings.append(
                f"Cannot load raster data for: {', '.join(missing)}."
            )
            trace.append(self._trace_dict(
                "PHASE_4_COMPLETED", "",
                {"status": "failed", "reason": "raster_load_error"},
            ))
            return ChangePipelineResult(
                analysis_id=analysis_id,
                result=ChangeUnderstandingResult(
                    analysis_id=analysis_id,
                    warnings=warnings,
                    summary={"total_regions": 0, "change_type_counts": {},
                             "total_changed_pixels": 0, "confidence_summary": {},
                             "warning_count": len(warnings)},
                ),
                models_used=self._models_metadata(),
                pipeline_trace=trace,
                warnings=warnings,
            )

        trace.append(self._trace_dict(
            "PHASE_4_STARTED", "",
            {"t1_shape": list(t1_pixels.shape),
             "t2_shape": list(t2_pixels.shape),
             "load_ms": round(load_ms, 2)},
        ))

        # ── Adapt Phase 3 regions ────────────────────────────────
        adapted = [
            ChangeRegionInput.from_phase3_change_region(cr)
            for cr in change_regions
        ]

        # ── Merge band descriptions ──────────────────────────────
        # Prefer T1 descriptions; if absent try T2.
        descs = t1_descs if t1_descs else t2_descs
        interps = t1_interps if t1_interps else t2_interps

        resolution = t1_asset.resolution if t1_asset.resolution else None

        # ── Run understander ─────────────────────────────────────
        trace.append(self._trace_dict("LAND_COVER_CLASSIFICATION", "", {}))
        trace.append(self._trace_dict("CHANGE_UNDERSTANDING", "", {}))

        t0 = time.time_ns()
        result = self._understander.understand_changes(
            t1_pixels,
            t2_pixels,
            adapted,
            band_descriptions=descs,
            color_interpretations=interps,
            resolution=resolution,
        )
        pipeline_ms = (time.time_ns() - t0) / 1e6

        # Override analysis_id to match pipeline.
        result.analysis_id = analysis_id

        trace.append(self._trace_dict(
            "CONFIDENCE_CALCULATION", "",
            {"mean_confidence": result.summary.get("confidence_summary", {}).get("mean")},
        ))

        trace.append(self._trace_dict(
            "PHASE_4_COMPLETED", "",
            {"status": "success",
             "total_regions": result.summary.get("total_regions", 0),
             "pipeline_ms": round(pipeline_ms, 2)},
        ))

        # ── Run Phase 5 Evidence Extractor ────────────────────────
        trace.append(self._trace_dict("PHASE_5_STARTED", "", {}))
        trace.append(self._trace_dict("EVIDENCE_EXTRACTION_STARTED", "", {}))
        
        t1_date = t1_asset.timestamp.isoformat() if t1_asset.timestamp else None
        t2_date = t2_asset.timestamp.isoformat() if t2_asset.timestamp else None
        
        t0 = time.time_ns()
        evidence_results = self._extractor.extract(
            t1_pixels,
            t2_pixels,
            adapted,
            result.regions,
            t1_date=t1_date,
            t2_date=t2_date
        )
        phase5_ms = (time.time_ns() - t0) / 1e6
        
        trace.append(self._trace_dict("EVIDENCE_RESULTS_ATTACHED", "", {}))
        trace.append(self._trace_dict("PHASE_5_COMPLETED", "", {"pipeline_ms": round(phase5_ms, 2)}))

        return ChangePipelineResult(
            analysis_id=analysis_id,
            result=result,
            models_used=self._models_metadata(),
            pipeline_trace=trace,
            warnings=warnings,
            evidence_results=evidence_results,
        )

    # ── Array-Level (for tests / direct use) ─────────────────────

    def run_pipeline_from_arrays(
        self,
        t1_pixels: np.ndarray,
        t2_pixels: np.ndarray,
        change_regions: List[ChangeRegionInput],
        *,
        band_descriptions: Optional[List[Optional[str]]] = None,
        color_interpretations: Optional[List[Optional[str]]] = None,
        resolution: Optional[List[float]] = None,
    ) -> ChangePipelineResult:
        """Run the pipeline from pre-loaded arrays.

        This bypasses raster I/O and is intended for testing and
        direct programmatic use.
        """
        analysis_id = str(uuid.uuid4())
        trace: List[Dict[str, Any]] = []

        trace.append(self._trace_dict(
            "PHASE_3_COMPLETED", "",
            {"change_regions_count": len(change_regions)},
        ))
        trace.append(self._trace_dict(
            "CHANGE_REGIONS_AVAILABLE", "",
            {"region_ids": [cr.region_id for cr in change_regions]},
        ))
        trace.append(self._trace_dict("PHASE_4_STARTED", "", {
            "t1_shape": list(t1_pixels.shape),
            "t2_shape": list(t2_pixels.shape),
        }))
        trace.append(self._trace_dict("LAND_COVER_CLASSIFICATION", "", {}))
        trace.append(self._trace_dict("CHANGE_UNDERSTANDING", "", {}))

        t0 = time.time_ns()
        result = self._understander.understand_changes(
            t1_pixels,
            t2_pixels,
            change_regions,
            band_descriptions=band_descriptions,
            color_interpretations=color_interpretations,
            resolution=resolution,
        )
        pipeline_ms = (time.time_ns() - t0) / 1e6
        result.analysis_id = analysis_id

        trace.append(self._trace_dict(
            "CONFIDENCE_CALCULATION", "",
            {"mean_confidence": result.summary.get("confidence_summary", {}).get("mean")},
        ))
        trace.append(self._trace_dict(
            "PHASE_4_COMPLETED", "",
            {"status": "success",
             "total_regions": result.summary.get("total_regions", 0),
             "pipeline_ms": round(pipeline_ms, 2)},
        ))
        
        # ── Run Phase 5 Evidence Extractor ────────────────────────
        trace.append(self._trace_dict("PHASE_5_STARTED", "", {}))
        trace.append(self._trace_dict("EVIDENCE_EXTRACTION_STARTED", "", {}))
        
        print("DEBUG PHASE 4 RESULT REGIONS:", [r.spectral_features_t1 for r in result.regions])
        
        t0 = time.time_ns()
        evidence_results = self._extractor.extract(
            t1_pixels,
            t2_pixels,
            change_regions,
            result.regions,
        )
        phase5_ms = (time.time_ns() - t0) / 1e6
        
        trace.append(self._trace_dict("EVIDENCE_RESULTS_ATTACHED", "", {}))
        trace.append(self._trace_dict("PHASE_5_COMPLETED", "", {"pipeline_ms": round(phase5_ms, 2)}))

        return ChangePipelineResult(
            analysis_id=analysis_id,
            result=result,
            models_used=self._models_metadata(),
            pipeline_trace=trace,
            warnings=[],
            evidence_results=evidence_results,
        )

    # ── Raster Loading ───────────────────────────────────────────

    @staticmethod
    def _load_raster(
        asset: RasterAsset,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[List[Optional[str]]],
        Optional[List[Optional[str]]],
        List[str],
    ]:
        """Load raster pixels from an asset.

        Returns ``(pixels, band_descriptions, color_interpretations, warnings)``.
        ``pixels`` is ``None`` when the file cannot be read.
        """
        warnings: List[str] = []
        file_path = Path(asset.path)
        if not file_path.is_absolute():
            file_path = settings.base_dir / asset.path

        if not file_path.exists():
            warnings.append(f"Raster file not found: {file_path}")
            return None, None, None, warnings

        try:
            with rasterio.open(file_path) as src:
                pixels = src.read().astype(np.float64)
                descs = list(src.descriptions) if src.descriptions else None
                interps = None
                if hasattr(src, "colorinterp") and src.colorinterp:
                    interps = [
                        ci.name.lower() if ci else None
                        for ci in src.colorinterp
                    ]

                # SAR guard.
                if asset.modality == "sar":
                    warnings.append(
                        "SAR imagery detected. Optical spectral indices "
                        "will not be meaningful."
                    )

                return pixels, descs, interps, warnings
        except Exception as e:
            warnings.append(f"Failed to read raster: {e}")
            return None, None, None, warnings

    # ── Helpers ──────────────────────────────────────────────────

    def _models_metadata(self) -> List[Dict[str, str]]:
        return [
            {
                "model_id": "spectral_land_cover_classifier",
                "model_name": self._classifier.classifier_name,
                "version": self._classifier.classifier_version,
                "task": "LAND_COVER_CLASSIFICATION",
                "type": "spectral_baseline",
            },
            {
                "model_id": "rule_based_change_understander",
                "model_name": self._understander.understander_name,
                "version": self._understander.understander_version,
                "task": "CHANGE_UNDERSTANDING",
                "type": "rule_based_baseline",
            },
        ]

    @staticmethod
    def _trace_dict(
        event: str,
        region_id: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "event": event,
            "region_id": region_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }


# Module-level singleton.
change_pipeline = ChangePipelineService()
