from fastapi import APIRouter, HTTPException
from app.services.pipeline_orchestrator import orchestrator, OrchestratorRequest

router = APIRouter()

@router.post("/orchestrate")
async def orchestrate_pipeline(request: OrchestratorRequest):
    """
    Agentic Orchestrator endpoint.
    Translates natural language queries into deterministic pipelines, validates inputs,
    and returns a structured execution trace with evidence-grounded results.
    """
    try:
        response = orchestrator.execute(request)
        # Even if it's an error (e.g., UNAVAILABLE or NEEDS_CLARIFICATION), 
        # we return it as a structured 200 OK so the frontend agent can understand the context.
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
