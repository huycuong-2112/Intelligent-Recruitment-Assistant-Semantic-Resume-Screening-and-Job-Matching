from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

class ExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match_run_id: str
    cv_id: str
    mode: Literal["auto", "offline", "groq"] = "auto"

class NarrativeItem(BaseModel):
    text: str
    evidence_refs: list[str] = Field(default_factory=list)
    type: str | None = None
    fact_id: str | None = None

class NarrativeGap(NarrativeItem):
    type: str = "other"

class NarrativeInterview(BaseModel):
    topic: str
    question: str
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)
    fact_id: str | None = None

class ExplanationNarrative(BaseModel):
    summary: str
    strengths: list[NarrativeItem] = Field(default_factory=list)
    gaps: list[NarrativeGap] = Field(default_factory=list)
    interview_focus: list[NarrativeInterview] = Field(default_factory=list)
    disclaimer: str

class ExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = "explanation_v1"
    source_xai_schema_version: str = "xai_v1"
    match_run_id: str
    cv_id: str
    jd_id: str
    target_role: str | None = None
    scoring_model_version: str | None = None
    decision: dict[str, Any]
    generation: dict[str, Any]
    explanation: ExplanationNarrative
