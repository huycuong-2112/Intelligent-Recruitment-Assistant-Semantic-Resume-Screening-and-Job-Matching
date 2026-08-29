from fastapi import APIRouter, HTTPException
from app.api.schemas.explanation_schema import ExplanationRequest, ExplanationResponse
from app.api.services.runtime_explanation_service import generate_runtime_explanation
from app.api.services.runtime_xai_service import RuntimeXAIError

router = APIRouter(prefix="/explanations", tags=["Explanations"])

@router.post("/generate", response_model=ExplanationResponse)
def generate(request: ExplanationRequest):
    try:
        return generate_runtime_explanation(request.match_run_id, request.cv_id, request.mode)
    except (RuntimeXAIError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
