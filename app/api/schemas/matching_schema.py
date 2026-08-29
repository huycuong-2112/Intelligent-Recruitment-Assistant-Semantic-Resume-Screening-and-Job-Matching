from pydantic import BaseModel, Field
from typing import List

class RuntimeDocumentRef(BaseModel):
    run_id: str
    document_id: str

class RuntimeMatchRequest(BaseModel):
    domain: str
    job: RuntimeDocumentRef
    candidates: List[RuntimeDocumentRef] = Field(..., min_length=1)

class RuntimeMatchResponse(BaseModel):
    match_run_id: str
    domain: str
    job: RuntimeDocumentRef
    results: list[dict]
    manifest: dict
