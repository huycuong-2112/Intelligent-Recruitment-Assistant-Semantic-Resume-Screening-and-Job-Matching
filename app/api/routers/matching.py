from fastapi import APIRouter
from app.api.schemas.matching_schema import RuntimeMatchRequest, RuntimeMatchResponse
from app.api.services.runtime_matching_service import run_runtime_matching, RuntimeMatchingError
from fastapi import HTTPException

router = APIRouter(prefix="/matching", tags=["Matching"])

@router.post("/run", response_model=RuntimeMatchResponse)
def run_matching(request: RuntimeMatchRequest):
    try:
        return run_runtime_matching(request.domain, request.job.model_dump(), [x.model_dump() for x in request.candidates])
    except RuntimeMatchingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
