import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

from app.schemas import AnalysisResponse, EvidenceResult, RasterAsset, TraceNode, ModelInfo
from app.services.model_registry import model_registry, BaseModelAdapter, VQAModelAdapter, CaptionModelAdapter, GroundingModelAdapter
from app.services.raster_ingestion_service import raster_ingestion_service


class AgentService:
    """Specialist Controller routing user queries to registry models, extracting evidence, and building execution traces."""

    def __init__(self):
        # Maps (image_id, session_id) -> list of turns: [{"query": str, "answer": str}]
        self.conversations: Dict[Tuple[str, str], List[Dict[str, str]]] = {}

    def normalize_query(self, query: str) -> str:
        """Clean and normalize query, handling capitalization, punctuation, and wording variations."""
        q = query.lower().strip()
        q = q.rstrip("?.!").lstrip()
        
        # Normalize water references
        if "any water bodies" in q or "see water" in q or "water body" in q or "water" in q:
            if q in ["can you see water", "are there any water bodies", "is there a water body", "is there water body"]:
                q = "is there water"
            else:
                q = q.replace("any water bodies", "water").replace("see water", "water").replace("water body", "water")
            
        # Normalize agricultural references
        if "agricultural fields" in q or "agricultural plots" in q or "agricultural land" in q:
            if q in ["identify agricultural plots", "identify agricultural fields"]:
                q = "identify agricultural land"
            else:
                q = q.replace("agricultural fields", "agricultural land").replace("agricultural plots", "agricultural land")
            
        # Normalize building references
        if "any buildings" in q or "see buildings" in q or "structures" in q:
            q = q.replace("any buildings", "buildings").replace("see buildings", "buildings").replace("structures", "buildings")
            
        return q

    def classify_task(self, query: str) -> str:
        """Classify user query into SINGLE_VQA, CAPTIONING, or GROUNDING."""
        q = query.lower().strip()
        
        # 1. Captioning/Scene description keywords
        caption_keywords = ["describe", "caption", "scene description", "overview", "summarize", "summary"]
        if any(k in q for k in caption_keywords):
            return "CAPTIONING"
            
        # 2. Grounding keywords
        ground_keywords = ["highlight", "locate", "find", "ground", "where is", "bbox", "bounding box", "mask", "segment"]
        if any(k in q for k in ground_keywords):
            return "GROUNDING"
            
        # 3. Default
        return "SINGLE_VQA"

    def determine_confidence_level(self, score: float) -> str:
        """Categorize numerical confidence score into HIGH, MEDIUM, LOW."""
        if score >= 0.85:
            return "HIGH"
        elif score >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"

    def run_analysis(
        self, 
        image_id: str, 
        query: str, 
        preferred_model_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AnalysisResponse:
        """Route user query, execute model inference, build evidence, and return final standardized AnalysisResponse."""
        start_time_ns = time.time_ns()
        analysis_id = str(uuid.uuid4())
        
        # 1. Fetch raster asset
        asset = raster_ingestion_service.get_asset(image_id)
        if not asset:
            raise FileNotFoundError(f"Raster asset {image_id} not found.")

        # 2. Query Normalization & Conversation Memory Resolution
        normalized_query = self.normalize_query(query)
        resolved_query = query
        
        if session_id:
            key = (image_id, session_id)
            if key not in self.conversations:
                self.conversations[key] = []
            
            history = self.conversations[key]
            
            # Contextual pronoun resolution (e.g. "where is it", "highlight it", "locate it")
            if normalized_query in ["where is it", "locate it", "highlight it", "show it", "where"]:
                # Look back in history for the last question containing a known entity
                for turn in reversed(history):
                    past_q = turn.get("query", "").lower()
                    if "water" in past_q:
                        resolved_query = "Highlight the water body"
                        break
                    elif "road" in past_q:
                        resolved_query = "Highlight the road"
                        break
                    elif "building" in past_q or "house" in past_q or "structure" in past_q:
                        resolved_query = "Highlight buildings"
                        break
                    elif "field" in past_q or "crop" in past_q or "agriculture" in past_q:
                        resolved_query = "Highlight agricultural regions"
                        break
                normalized_query = self.normalize_query(resolved_query)

        # 3. Classify task using resolved query
        task = self.classify_task(resolved_query)
        
        # 4. Route to model
        # Choose default model based on task and modality if not specified
        if not preferred_model_id:
            if task == "SINGLE_VQA":
                if asset.modality == "sar":
                    # Generic VLM or Demo Fallback since RS VQA Adapted only supports optical
                    preferred_model_id = "demo_fallback"
                else:
                    preferred_model_id = "rs_vqa_adapted"
            else:
                # Captioning/Grounding adapters are integrated in demo_fallback for this baseline phase
                preferred_model_id = "demo_fallback"

        warnings = []
        errors = []
        
        model_adapter = model_registry.get_model(preferred_model_id)
        if model_adapter:
            model_info = model_adapter.get_info()
            if asset.modality not in model_info.supported_modalities:
                warnings.append(f"{model_info.name} does not currently support {asset.modality} modality. Falling back to Demo Fallback model.")
                preferred_model_id = "demo_fallback"
                model_adapter = model_registry.get_model(preferred_model_id)

        if not model_adapter:
            # Fallback to demo mode
            preferred_model_id = "demo_fallback"
            model_adapter = model_registry.get_model(preferred_model_id)

        model_info = model_adapter.get_info()
        evidence: List[EvidenceResult] = []
        answer = ""
        confidence_score = 0.5
        
        # Keep execution trace nodes
        trace_nodes: List[TraceNode] = []
        
        # Node 1: Task Classification
        t1_start = datetime.now(timezone.utc)
        trace_nodes.append(TraceNode(
            step_id="task_classification",
            model_name="SatQuery Router Classifier",
            display_name="Query Task Classification",
            status="success",
            started_at=t1_start,
            completed_at=datetime.now(timezone.utc),
            duration_ms=0.1,
            inputs_summary={"query": query, "resolved_query": resolved_query},
            outputs_summary={"classified_task": task}
        ))

        # Node 2: Modality & Compatibility check
        t2_start = datetime.now(timezone.utc)
        modality_supported = asset.modality in model_info.supported_modalities
        if not modality_supported:
            warnings.append(f"Modality '{asset.modality}' is not natively supported by model '{model_info.name}'. Fallback behavior engaged.")
            
        trace_nodes.append(TraceNode(
            step_id="modality_check",
            model_name="SatQuery Constraint Engine",
            display_name="Input Modality Compatibility Check",
            status="success" if modality_supported else "warning",
            started_at=t2_start,
            completed_at=datetime.now(timezone.utc),
            duration_ms=0.1,
            inputs_summary={"asset_modality": asset.modality, "model_supported_modalities": model_info.supported_modalities},
            outputs_summary={"compatible": modality_supported}
        ))

        # Node 3: Specialist Model Inference
        t3_start = datetime.now(timezone.utc)
        inf_duration_ms = 0.0
        
        try:
            if task == "SINGLE_VQA":
                if not isinstance(model_adapter, VQAModelAdapter):
                    # Route to demo fallback
                    model_adapter = model_registry.get_model("demo_fallback")
                    model_info = model_adapter.get_info()
                
                ans, conf, ev, warns = model_adapter.answer_question(asset, query)
                answer = ans
                confidence_score = conf
                evidence.extend(ev)
                warnings.extend(warns)
                
            elif task == "CAPTIONING":
                if not isinstance(model_adapter, CaptionModelAdapter):
                    model_adapter = model_registry.get_model("demo_fallback")
                    model_info = model_adapter.get_info()
                    
                caption_dict, conf, ev, warns = model_adapter.generate_caption(asset)
                # Format structured caption as structured text response
                answer = "\n\n".join([f"### {k}\n{v}" for k, v in caption_dict.items()])
                confidence_score = conf
                evidence.extend(ev)
                warnings.extend(warns)
                
            elif task == "GROUNDING":
                if not isinstance(model_adapter, GroundingModelAdapter):
                    model_adapter = model_registry.get_model("demo_fallback")
                    model_info = model_adapter.get_info()
                    
                ev, conf, warns = model_adapter.ground_text(asset, query)
                
                if len(ev) > 0 and ev[0].bbox:
                    bbox_str = ", ".join([f"{c:.4f}" for c in ev[0].bbox])
                    answer = f"Successfully localized '{query}'. Bounding region: [{bbox_str}]."
                else:
                    answer = f"No localized region matches query: '{query}'."
                    
                confidence_score = conf
                evidence.extend(ev)
                warnings.extend(warns)
                
            inf_duration_ms = (time.time_ns() - start_time_ns) / 1e6
            
            trace_nodes.append(TraceNode(
                step_id="model_inference",
                model_name=model_info.name,
                display_name="Specialist Model Inference Execution",
                status="success",
                started_at=t3_start,
                completed_at=datetime.now(timezone.utc),
                duration_ms=inf_duration_ms,
                inputs_summary={"query": query, "model_id": model_info.model_id},
                outputs_summary={"raw_response": answer[:100] + "..." if len(answer) > 100 else answer, "confidence": confidence_score}
            ))
            
        except Exception as e:
            answer = "No compatible remote-sensing model is currently available for this task."
            confidence_score = 0.0
            inf_duration_ms = (time.time_ns() - start_time_ns) / 1e6
            warnings.append(f"Model inference failed: {str(e)}")
            
            trace_nodes.append(TraceNode(
                step_id="model_inference",
                model_name=model_info.name,
                display_name="Specialist Model Inference Execution",
                status="error",
                started_at=t3_start,
                completed_at=datetime.now(timezone.utc),
                duration_ms=inf_duration_ms,
                inputs_summary={"query": query},
                error=str(e)
            ))

        # Node 4: Evidence Synthesis
        t4_start = datetime.now(timezone.utc)
        trace_nodes.append(TraceNode(
            step_id="evidence_synthesis",
            model_name="SatQuery Evidence Engine",
            display_name="Evidence Synthesis & Labeled Certainty",
            status="success",
            started_at=t4_start,
            completed_at=datetime.now(timezone.utc),
            duration_ms=0.5,
            inputs_summary={"raw_evidence_count": len(evidence)},
            outputs_summary={"quantified_evidence": [e.model_dump() for e in evidence]}
        ))

        # Prepare confidence mapping
        confidence_level = self.determine_confidence_level(confidence_score)
        confidence_map = {
            "level": confidence_level,
            "score": confidence_score,
            "type": "Model confidence"
        }

        # Formulate response
        response = AnalysisResponse(
            analysis_id=analysis_id,
            task=task,
            answer=answer,
            evidence=evidence,
            confidence=confidence_map,
            model=model_info,
            warnings=warnings,
            execution_trace=trace_nodes
        )
        
        # Save to conversation memory if session_id exists
        if session_id:
            key = (image_id, session_id)
            if key not in self.conversations:
                self.conversations[key] = []
            self.conversations[key].append({
                "query": query,
                "answer": answer
            })
        
        return response


agent_service = AgentService()
