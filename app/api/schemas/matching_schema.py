from pydantic import BaseModel, Field
from typing import List

class MatchRequest(BaseModel):
    resume_text: str = Field(..., description="Nội dung CV đã trích xuất")
    job_description: str = Field(..., description="Nội dung mô tả công việc")

class MatchResponse(BaseModel):
    similarity_score: float = Field(..., ge=0, le=1, description="Điểm tương đồng 0-1")
    matched_skills: List[str] = []
    explanation: str = ""