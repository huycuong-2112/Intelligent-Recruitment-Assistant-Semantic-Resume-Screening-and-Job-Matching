from fastapi import APIRouter
from app.api.schemas.matching_schema import MatchRequest, MatchResponse
from app.api.services.matching_service import match_resume_to_job

router = APIRouter(prefix="/matching", tags=["Matching"])

@router.post("/score", response_model=MatchResponse)
def score_matching(request: MatchRequest):
    return match_resume_to_job(request)