from app.ml.inference.matcher import matcher
from app.api.schemas.matching_schema import MatchRequest, MatchResponse

def match_resume_to_job(request: MatchRequest) -> MatchResponse:
    score = matcher.compute_similarity(request.resume_text, request.job_description)
    return MatchResponse(
        similarity_score=score,
        matched_skills=[],
        explanation="Kết quả tạm thời (model chưa tích hợp)"
    )