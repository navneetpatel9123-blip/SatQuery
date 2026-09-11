# Phase 6: Agentic Pipeline Orchestrator

## Architecture
The Pipeline Orchestrator translates unconstrained natural language queries into deterministic, strictly executable machine workflows. It sits as the master entrypoint for the `/api/v1/orchestrate` endpoint, delegating internally to Phase 3, Phase 4, and Phase 5 modules based on query intent.

## Task Taxonomy
The orchestrator recognizes the following intents based on NLP keywords:
- `BI_TEMPORAL_CHANGE`: Requests basic anomaly/change detection maps.
- `BI_TEMPORAL_CHANGE_UNDERSTANDING`: Requests semantic categorization of detected changes.
- `BI_TEMPORAL_EVIDENCE`: Requests complete explanation and structured evidence for the pipeline's classifications.
- `SINGLE_IMAGE_VQA`: Visual Question Answering on one asset (Placeholder/Stub).
- `SINGLE_IMAGE_CAPTION`: Image captioning on one asset (Placeholder/Stub).
- `SINGLE_IMAGE_GROUNDING`: Bounding box generation from text (Placeholder/Stub).
- `OPTICAL_SAR_ANALYSIS`: Cross-modality fusion queries (Placeholder/Stub).
- `UNKNOWN_TASK`: Ambiguous or unrelated questions.

## Workflow Planning
Currently, the pipeline planner is strictly deterministic. Rather than letting an LLM generate arbitrary tools to run, the system maps `QueryTask` Enums directly to execution sequences:
- `BI_TEMPORAL_EVIDENCE` -> `["CHANGE_DETECTION", "CHANGE_UNDERSTANDING", "EVIDENCE_EXTRACTION"]`

The execution layer loops over the sequence, passing structured `dataclass` state sequentially between the Phase adapters.

## Input Validation & Security Restrictions
Before any models are run, the `InputValidator` checks the raster context against the chosen `QueryTask`:
- If `BI_TEMPORAL_*` is requested, it ensures exactly two rasters are loaded, preventing the pipeline from crashing deep within Phase 3.
- The planner **does not execute shell commands or arbitrary Python**. It restricts execution exclusively to models registered in `model_registry.py`.

## Unavailable Capabilities
If a user asks for `SINGLE_IMAGE_VQA`, the orchestrator identifies the task, attempts to construct a workflow, finds it unsupported, and returns `status="UNAVAILABLE"` seamlessly, preserving the application's uptime and gracefully degrading the UX.

## Future VLM/LLM Integration
The current query router is deterministic and transparent. Future iterations (e.g., Phase 9) may replace the `TaskClassifier` and `WorkflowPlanner` with an adapted LLM or VLM. The architecture guarantees that even if the planner is replaced, the identical safety boundaries, input validations, and capability registry checks will protect the system from hallucinated execution.

## Execution Trace
The orchestrator prepends pipeline-level routing telemetry into the same execution trace used in Phase 3/4/5:
`ORCHESTRATION_STARTED` → `QUERY_CLASSIFIED` → `INPUT_VALIDATED` → `WORKFLOW_PLANNED` → `STEP_STARTED` → `...` → `ORCHESTRATION_COMPLETED`.
