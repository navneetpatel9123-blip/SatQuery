# Agent Workflow & Execution Trace

The SatQuery specialist agent acts as the main orchestrator, interpreting natural language queries and routing them through a pipeline of constraint validators and specialist models.

## Pipeline Flow

1. **User Query Input**: The user enters a question about the active ingested asset (e.g. "Highlight the river").
2. **Query Interpretation & Classification**:
   - `AgentService` uses keyphrase heuristics to determine the specialist category: `SINGLE_VQA`, `CAPTIONING`, or `GROUNDING`.
3. **Modality & Constraint Validator**:
   - Checks if the selected model supports the raster modality (e.g. raises warning if running SAR on an optical-only model).
   - Automatically fallbacks to `demo_fallback` model to prevent crashes.
4. **Specialist Model Execution**:
   - Calls the model adapter's interface and computes answers and visual/numerical confidence scores.
5. **Evidence Extraction**:
   - Extracts bounding coordinates or patch statistics and labels their certainty (`OBSERVED`, `INFERRED`, or `UNCERTAIN`).
6. **DAG Trace Synthesis**:
   - Records elapsed processing time and execution status for each step and emits the `AnalysisResponse`.

## DAG Execution Nodes

- **`Query Task Classification`**: Identifies user intent.
- **`Input Modality Compatibility Check`**: Verifies grid modality constraints.
- **`Specialist Model Inference Execution`**: Executes the adapter model.
- **`Evidence Synthesis & Labeled Certainty`**: Quantifies supporting evidence.
- **`Response Synthesis`**: Returns the standardized payload.
