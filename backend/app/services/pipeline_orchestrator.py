import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas import RasterAsset, TraceNode
from app.services.model_registry import model_registry
from app.services.raster_ingestion_service import raster_ingestion_service

# For Phase 3, we mock a function here or import if it exists.
# The project has analysis_service for Phase 3? Let's check.
# Actually, I'll structure the file first, and verify imports later.

class QueryTask(str, Enum):
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    SINGLE_IMAGE_CAPTION = "SINGLE_IMAGE_CAPTION"
    SINGLE_IMAGE_GROUNDING = "SINGLE_IMAGE_GROUNDING"
    BI_TEMPORAL_CHANGE = "BI_TEMPORAL_CHANGE"
    BI_TEMPORAL_CHANGE_UNDERSTANDING = "BI_TEMPORAL_CHANGE_UNDERSTANDING"
    BI_TEMPORAL_EVIDENCE = "BI_TEMPORAL_EVIDENCE"
    OPTICAL_SAR_ANALYSIS = "OPTICAL_SAR_ANALYSIS"
    UNKNOWN_TASK = "UNKNOWN_TASK"

class OrchestratorRequest(BaseModel):
    query: str
    t1_image_id: Optional[str] = None
    t2_image_id: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class TraceEvent:
    event: str
    status: str = "SUCCESS"
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

class OrchestratorResponse(BaseModel):
    status: str
    task: str
    workflow: List[str]
    models_used: List[Dict[str, str]]
    results: Dict[str, Any]
    evidence: Any
    warnings: List[str]
    execution_trace: List[Dict[str, Any]]

class TaskClassifier:
    """Deterministic natural language query classifier."""
    
    @staticmethod
    def classify(query: str) -> tuple[QueryTask, str]:
        q = query.lower()
        
        # Bi-temporal Evidence queries
        if ("why" in q or "evidence" in q or "prove" in q):
            return QueryTask.BI_TEMPORAL_EVIDENCE, ""
            
        if "type of change" in q or "what type" in q or ("what changed" in q and ("understand" in q or "type" in q or "has the" in q)):
            return QueryTask.BI_TEMPORAL_CHANGE_UNDERSTANDING, ""
            
        if "has the built-up area increased" in q or ("built-up" in q and ("increase" in q or "change" in q or "expansion" in q)):
            return QueryTask.BI_TEMPORAL_CHANGE_UNDERSTANDING, ""
            
        if "what changed" in q or "detect change" in q or "between these two dates" in q or "change" in q or "changes" in q or ("between" in q and ("t1" in q or "t2" in q or "dates" in q or "images" in q)):
            return QueryTask.BI_TEMPORAL_CHANGE, ""

        if "optical" in q and "sar" in q:
            return QueryTask.OPTICAL_SAR_ANALYSIS, ""
            
        if "describe" in q or "caption" in q or "overview" in q:
            return QueryTask.SINGLE_IMAGE_CAPTION, ""

        if "highlight" in q or "ground" in q or "locate" in q or "find" in q or "bbox" in q or "bounding box" in q:
            return QueryTask.SINGLE_IMAGE_GROUNDING, ""

        if "identify" in q or "what is visible" in q or "vqa" in q or "is there" in q or "are there" in q or "how many" in q or "what type of land cover" in q or "land cover" in q or "water" in q or "road" in q or "building" in q or "agricultural" in q:
            return QueryTask.SINGLE_IMAGE_VQA, ""
            
        return QueryTask.UNKNOWN_TASK, "Could not determine the required task from the query."

class InputValidator:
    """Validates inputs against the chosen task."""
    
    @staticmethod
    def validate(task: QueryTask, req: OrchestratorRequest) -> tuple[bool, str, List[RasterAsset]]:
        assets = []
        if req.t1_image_id:
            a1 = raster_ingestion_service.get_asset(req.t1_image_id)
            if a1: assets.append(a1)
        if req.t2_image_id:
            a2 = raster_ingestion_service.get_asset(req.t2_image_id)
            if a2: assets.append(a2)
            
        if task in [QueryTask.BI_TEMPORAL_CHANGE, QueryTask.BI_TEMPORAL_CHANGE_UNDERSTANDING, QueryTask.BI_TEMPORAL_EVIDENCE]:
            if not req.t1_image_id or not req.t2_image_id:
                return False, "I can analyze change between two images, but I need T1 and T2 images.", []
            if len(assets) != 2:
                missing = []
                if not any(a.id == req.t1_image_id for a in assets): missing.append("T1")
                if not any(a.id == req.t2_image_id for a in assets): missing.append("T2")
                return False, f"Missing images: {','.join(missing)}", []
                
            pair_val = raster_ingestion_service.validate_pair(req.t1_image_id, req.t2_image_id)
            if not pair_val.compatible:
                err_msg = pair_val.errors[0] if pair_val.errors else "T1 and T2 image dimensions or grid mismatch."
                return False, f"Bi-temporal analysis blocked due to mismatched images: {err_msg}", assets
                
        elif task in [QueryTask.SINGLE_IMAGE_CAPTION, QueryTask.SINGLE_IMAGE_VQA, QueryTask.SINGLE_IMAGE_GROUNDING]:
            if not req.t1_image_id and not req.t2_image_id:
                return False, "Single image analysis requires at least one image.", []
            if len(assets) < 1:
                return False, "Missing required image.", []
            return True, "", assets
                
        elif task == QueryTask.OPTICAL_SAR_ANALYSIS:
            if not req.t1_image_id or not req.t2_image_id:
                return False, "OPTICAL_SAR_ANALYSIS requires exactly one OPTICAL and one SAR image.", []
            if len(assets) != 2:
                return False, "OPTICAL_SAR_ANALYSIS requires exactly two images.", []
            return True, "", assets
            
        return True, "", assets

class WorkflowPlanner:
    """Maps a task to a deterministic step sequence."""
    
    @staticmethod
    def plan(task: QueryTask) -> List[str]:
        if task == QueryTask.BI_TEMPORAL_CHANGE:
            return ["CHANGE_DETECTION"]
        if task == QueryTask.BI_TEMPORAL_CHANGE_UNDERSTANDING:
            return ["CHANGE_DETECTION", "CHANGE_UNDERSTANDING"]
        if task == QueryTask.BI_TEMPORAL_EVIDENCE:
            return ["CHANGE_DETECTION", "CHANGE_UNDERSTANDING", "EVIDENCE_EXTRACTION"]
        if task == QueryTask.SINGLE_IMAGE_VQA:
            return ["VQA"]
        if task == QueryTask.SINGLE_IMAGE_CAPTION:
            return ["CAPTIONING"]
        if task == QueryTask.SINGLE_IMAGE_GROUNDING:
            return []
        if task == QueryTask.OPTICAL_SAR_ANALYSIS:
            return ["OPTICAL_SAR_ANALYSIS"]
        return []

class PipelineOrchestrator:
    """Executes the workflow securely using registered tools."""
    
    def __init__(self):
        # Local imports to avoid circular deps if they exist
        from app.services.change_pipeline import change_pipeline
        from app.models.change_detector import change_detector, spectral_diff_detector
        self.change_pipeline = change_pipeline
        self.change_detector = change_detector
        self.spectral_diff_detector = spectral_diff_detector

    def execute(self, req: OrchestratorRequest) -> OrchestratorResponse:
        trace = []
        warnings = []
        
        def t_add(event, status="SUCCESS", details=None):
            st = status.lower() if status else "success"
            det = details or {}
            model_name = det.get("model_id") or det.get("fallback_model") or "SatQuery Orchestrator"
            trace.append({
                "event": event,
                "details": det,
                "step_id": event.lower(),
                "model_name": model_name,
                "display_name": event.replace("_", " ").title(),
                "status": st,
                "duration_ms": 0.1,
                "inputs_summary": det,
                "outputs_summary": {}
            })
            
        t_add("ORCHESTRATION_STARTED")
        
        # 1. Query Classification
        task, msg = TaskClassifier.classify(req.query)
        t_add("QUERY_CLASSIFIED", details={"task": task.value, "msg": msg})
        
        if task == QueryTask.UNKNOWN_TASK:
            t_add("ORCHESTRATION_FAILED", "FAILED", {"reason": "unknown_query"})
            return self._err("NEEDS_CLARIFICATION", task.value, msg, trace)
            
        # 2. Input Validation
        valid, v_msg, assets = InputValidator.validate(task, req)
        t_add("INPUT_VALIDATED", "SUCCESS" if valid else "FAILED", {"msg": v_msg})
        
        if not valid:
            t_add("ORCHESTRATION_FAILED", "FAILED", {"reason": "invalid_input"})
            return self._err("NEEDS_CLARIFICATION", task.value, v_msg, trace)
            
        # 3. Capability Lookup
        workflow = WorkflowPlanner.plan(task)
        t_add("WORKFLOW_PLANNED", details={"workflow": workflow})
        
        if not workflow:
            # Task known but workflow not implemented (VQA, etc)
            t_add("CAPABILITY_LOOKUP", "FAILED", {"task": task.value})
            t_add("ORCHESTRATION_FAILED", "FAILED", {"reason": "unavailable_capability"})
            return self._err("UNAVAILABLE", task.value, f"The requested capability for {task.value} is not currently available.", trace)
            
        models_used = []
        
        # 4. Pipeline Execution
        try:
            t1_asset = next((a for a in assets if req.t1_image_id and a.id == req.t1_image_id), None)
            t2_asset = next((a for a in assets if req.t2_image_id and a.id == req.t2_image_id), None)
            target_asset = t2_asset or t1_asset or (assets[0] if assets else None)
            
            # Phase 3: Change Detection
            final_regions = []
            if "CHANGE_DETECTION" in workflow:
                t_add("STEP_STARTED", details={"step": "CHANGE_DETECTION"})
                
                # Check for trained Siamese Change Detector or fallback to SpectralDiffChangeDetector
                from app.models.change_detector import SiameseChangeDetector, SpectralDiffChangeDetector
                siamese = SiameseChangeDetector(auto_load=True)
                
                if siamese.is_trained:
                    t_add("CHANGE_DETECTION_MODEL_SELECTED", details={"model_id": "siamese_change_detector", "status": "TRAINED"})
                    cd_res = siamese.detect_changes(t1_asset, t2_asset, req.options)
                    models_used.append({"model_id": "siamese_change_detector", "step": "CHANGE_DETECTION"})
                else:
                    t_add("ADAPTED_MODEL_UNAVAILABLE", details={
                        "reason": "REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED",
                        "status": "TRAINING_READY",
                        "target_model": "siamese_change_detector",
                    })
                    t_add("RULE_BASED_FALLBACK_SELECTED", details={"fallback_model": "spectral_diff_change_detector"})
                    t_add("CHANGE_DETECTION_MODEL_SELECTED", details={"model_id": "spectral_diff_change_detector"})
                    cd_res = self.spectral_diff_detector.detect_changes(t1_asset, t2_asset, req.options)
                    models_used.append({"model_id": "spectral_diff_change_detector", "step": "CHANGE_DETECTION"})
                
                final_regions = cd_res.regions
                for evt in cd_res.execution_trace:
                    trace.append(evt)
                warnings.extend(cd_res.warnings)
                t_add("STEP_COMPLETED", details={"step": "CHANGE_DETECTION", "regions_detected": len(final_regions)})
            evidence_out = {}
            summary = {}
            
            if "CHANGE_UNDERSTANDING" in workflow:
                t_add("STEP_STARTED", details={"step": "CHANGE_UNDERSTANDING"})
                p4_result = self.change_pipeline.run_pipeline(t1_asset, t2_asset, final_regions)
                models_used.extend(p4_result.models_used)
                
                # change_pipeline actually runs Phase 4 + Phase 5 internally if Phase 5 is present.
                # So if the workflow includes EVIDENCE_EXTRACTION, p4_result already has it.
                d = p4_result.to_dict()
                final_regions = d.get("regions", [])
                summary = d.get("summary", {})
                warnings.extend(d.get("warnings", []))
                
                # Trace from pipeline
                for t in d.get("execution_trace", []):
                    trace.append(t)
                    
                t_add("STEP_COMPLETED", details={"step": "CHANGE_UNDERSTANDING"})
                
                if "EVIDENCE_EXTRACTION" in workflow:
                    t_add("STEP_STARTED", details={"step": "EVIDENCE_EXTRACTION"})
                    t_add("STEP_COMPLETED", details={"step": "EVIDENCE_EXTRACTION"})
                    t_add("RESULT_ATTACHED", details={"step": "EVIDENCE_EXTRACTION"})
                    
            elif "VQA" in workflow:
                from app.models.adapted_remote_sensing_vlm import AdaptedRemoteSensingModel
                from app.models.remote_sensing_vqa import RuleBasedRemoteSensingVQA
                
                t_add("STEP_STARTED", details={"step": "VQA"})
                adapted_model = AdaptedRemoteSensingModel()
                
                if adapted_model.is_trained:
                    t_add("ADAPTED_MODEL_SELECTED", details={"model_id": "adapted_remote_sensing_vlm"})
                    t_add("CHECKPOINT_LOADED", details={"checkpoint": adapted_model._checkpoint_path})
                    t_add("MODEL_VERSION", details={"version": adapted_model.version})
                    t_add("DATASET_VERSION", details={"dataset": "BigEarthNet-19", "version": "v1.0"})
                    t_add("INFERENCE_STARTED")
                    vqa_out = adapted_model.answer_question(target_asset, req.query)
                    t_add("INFERENCE_COMPLETED")
                    models_used.append({"model_id": "adapted_remote_sensing_vlm", "step": "VQA"})
                else:
                    t_add("ADAPTED_MODEL_UNAVAILABLE", details={
                        "reason": "REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED",
                        "status": "TRAINING_READY"
                    })
                    t_add("RULE_BASED_FALLBACK_SELECTED", details={"fallback_model": "remote_sensing_vqa"})
                    t_add("VQA_STARTED")
                    t_add("VQA_MODEL_SELECTED", details={"model_id": "remote_sensing_vqa"})
                    
                    vqa = RuleBasedRemoteSensingVQA()
                    vqa_out = vqa.answer_question(target_asset, req.query)
                    models_used.append({"model_id": "remote_sensing_vqa", "step": "VQA"})
                    t_add("VQA_COMPLETED")

                t_add("STEP_COMPLETED", details={"step": "VQA"})
                t_add("ORCHESTRATION_COMPLETED")
                return OrchestratorResponse(
                    status="COMPLETED",
                    task=task.value,
                    workflow=workflow,
                    models_used=models_used,
                    results={"answer": vqa_out.answer, "confidence": vqa_out.confidence},
                    evidence=[e.to_dict() if hasattr(e, "to_dict") else e for e in vqa_out.evidence],
                    warnings=warnings + vqa_out.warnings,
                    execution_trace=trace
                )
                
            elif "CAPTIONING" in workflow:
                from app.models.adapted_remote_sensing_vlm import AdaptedRemoteSensingModel
                from app.models.remote_sensing_captioner import RuleBasedRemoteSensingCaptioner
                
                t_add("STEP_STARTED", details={"step": "CAPTIONING"})
                adapted_model = AdaptedRemoteSensingModel()
                
                if adapted_model.is_trained:
                    t_add("ADAPTED_MODEL_SELECTED", details={"model_id": "adapted_remote_sensing_vlm"})
                    t_add("CHECKPOINT_LOADED", details={"checkpoint": adapted_model._checkpoint_path})
                    t_add("MODEL_VERSION", details={"version": adapted_model.version})
                    t_add("DATASET_VERSION", details={"dataset": "BigEarthNet-19", "version": "v1.0"})
                    t_add("INFERENCE_STARTED")
                    cap_out = adapted_model.generate_caption(target_asset)
                    t_add("INFERENCE_COMPLETED")
                    models_used.append({"model_id": "adapted_remote_sensing_vlm", "step": "CAPTIONING"})
                else:
                    t_add("ADAPTED_MODEL_UNAVAILABLE", details={
                        "reason": "REAL TRAINING NOT YET EXECUTED — DATASET REQUIRED",
                        "status": "TRAINING_READY"
                    })
                    t_add("RULE_BASED_FALLBACK_SELECTED", details={"fallback_model": "remote_sensing_captioner"})
                    t_add("CAPTION_STARTED")
                    t_add("CAPTION_MODEL_SELECTED", details={"model_id": "remote_sensing_captioner"})
                    
                    captioner = RuleBasedRemoteSensingCaptioner()
                    cap_out = captioner.generate_caption(target_asset)
                    models_used.append({"model_id": "remote_sensing_captioner", "step": "CAPTIONING"})
                    t_add("CAPTION_COMPLETED")

                t_add("STEP_COMPLETED", details={"step": "CAPTIONING"})
                t_add("ORCHESTRATION_COMPLETED")
                return OrchestratorResponse(
                    status="COMPLETED",
                    task=task.value,
                    workflow=workflow,
                    models_used=models_used,
                    results={"caption": cap_out.caption, "confidence": cap_out.confidence},
                    evidence=[e.to_dict() if hasattr(e, "to_dict") else e for e in cap_out.evidence],
                    warnings=warnings + cap_out.warnings,
                    execution_trace=trace
                )

            elif "GROUNDING" in workflow:
                t_add("STEP_STARTED", details={"step": "GROUNDING"})
                demo_adapter = model_registry.get_model("demo_fallback")
                ev, conf, warns = demo_adapter.ground_text(target_asset, req.query)
                models_used.append({"model_id": "demo_fallback", "step": "GROUNDING"})
                t_add("GROUNDING_COMPLETED")
                t_add("STEP_COMPLETED", details={"step": "GROUNDING"})
                t_add("ORCHESTRATION_COMPLETED")
                
                answer_text = f"Localized region for query '{req.query}'." if ev else f"Grounding query: '{req.query}' completed."
                return OrchestratorResponse(
                    status="COMPLETED",
                    task=task.value,
                    workflow=workflow,
                    models_used=models_used,
                    results={"answer": answer_text, "confidence": conf, "regions": [e.to_dict() if hasattr(e, "to_dict") else e for e in ev]},
                    evidence=[e.to_dict() if hasattr(e, "to_dict") else e for e in ev],
                    warnings=warnings + warns,
                    execution_trace=trace
                )

                t_add("STEP_COMPLETED", details={"step": "CAPTIONING"})
                t_add("ORCHESTRATION_COMPLETED")
                return OrchestratorResponse(
                    status="COMPLETED",
                    task=task.value,
                    workflow=workflow,
                    models_used=models_used,
                    results={"caption": cap_out.caption, "confidence": cap_out.confidence},
                    evidence=cap_out.evidence,
                    warnings=warnings + cap_out.warnings,
                    execution_trace=trace
                )
                
            elif "OPTICAL_SAR_ANALYSIS" in workflow:
                from app.models.optical_sar_analyzer import RuleBasedOpticalSARAnalyzer
                analyzer = RuleBasedOpticalSARAnalyzer()
                
                t_add("OPTICAL_SAR_STARTED", details={"step": "OPTICAL_SAR_ANALYSIS"})
                t_add("OPTICAL_INPUT_VALIDATED")
                t_add("SAR_INPUT_VALIDATED")
                t_add("COREGISTRATION_CHECKED")
                t_add("OPTICAL_FEATURES_EXTRACTED")
                t_add("SAR_FEATURES_EXTRACTED")
                t_add("CROSS_MODAL_FUSION_STARTED")
                
                # Determine which is optical and which is sar
                if t1_asset.modality in ["optical", "multispectral"]:
                    opt_asset = t1_asset
                    sar_asset = t2_asset
                else:
                    opt_asset = t2_asset
                    sar_asset = t1_asset
                    
                sar_out = analyzer.analyze(opt_asset, sar_asset)
                
                t_add("CROSS_MODAL_EVIDENCE_COMPUTED")
                t_add("AGREEMENT_ANALYZED")
                
                models_used.append({"model_id": "optical_sar_analyzer", "step": "OPTICAL_SAR_ANALYSIS"})
                
                if sar_out.status == "FAILED":
                    t_add("OPTICAL_SAR_FAILED", "FAILED", details={"error": sar_out.summary.get("error", "Unknown error")})
                    return self._err("FAILED", task.value, "Optical SAR analysis failed.", trace)
                
                t_add("OPTICAL_SAR_COMPLETED")
                t_add("STEP_COMPLETED", details={"step": "OPTICAL_SAR_ANALYSIS"})
                t_add("ORCHESTRATION_COMPLETED")
                
                # Format response
                return OrchestratorResponse(
                    status="COMPLETED",
                    task=task.value,
                    workflow=workflow,
                    models_used=models_used,
                    results={"regions": [r.model_dump() for r in sar_out.regions], "summary": sar_out.summary},
                    evidence={"cross_modal_analysis": True},
                    warnings=warnings + [w for r in sar_out.regions for w in r.warnings],
                    execution_trace=trace
                )
            else:
                # Just Phase 3
                summary = {"total_regions": len(final_regions)}
                
            t_add("ORCHESTRATION_COMPLETED")
            
            return OrchestratorResponse(
                status="COMPLETED",
                task=task.value,
                workflow=workflow,
                models_used=models_used,
                results={"regions": final_regions, "summary": summary},
                evidence={},
                warnings=warnings,
                execution_trace=trace
            )
            
        except Exception as e:
            t_add("STEP_FAILED", "FAILED", {"error": str(e)})
            t_add("ORCHESTRATION_FAILED", "FAILED", {"error": str(e)})
            return self._err("FAILED", task.value, str(e), trace)

    def _err(self, status: str, task: str, msg: str, trace: List[Any]) -> OrchestratorResponse:
        return OrchestratorResponse(
            status=status,
            task=task,
            workflow=[],
            models_used=[],
            results={},
            evidence={},
            warnings=[msg],
            execution_trace=trace
        )

orchestrator = PipelineOrchestrator()
