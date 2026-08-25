from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    MATCHED = "MATCHED"
    NO_EVIDENCE = "NO_EVIDENCE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")
    evidence_id: str
    source_type: str
    source_id: str | None = None
    source_text: str | None = None
    matcher: str | None = None
    score: float | None = None
    similarity: float | None = None


class Dimension(BaseModel):
    model_config = ConfigDict(extra="allow")
    score: float | None = Field(default=None, ge=0, le=1)
    status: EvidenceStatus
    weight: float = Field(ge=0, le=1)
    weighted_contribution: float | None = None
    coverage: float | None = Field(default=None, ge=0, le=1)


class Decision(BaseModel):
    final_score: float | None = Field(default=None, ge=0, le=1)
    status: str
    weights: dict[str, float]
    coverage: float | None = Field(default=None, ge=0, le=1)

    @field_validator("weights")
    @classmethod
    def weights_sum_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != {"skill", "experience", "education", "semantic"}:
            raise ValueError("weights must contain all four MDMS dimensions")
        if any(v < 0 for v in value.values()) or abs(sum(value.values()) - 1) > 1e-6:
            raise ValueError("weights must be non-negative and sum to 1")
        return value


class XAIOutput(BaseModel):
    """Canonical deterministic evidence object; no LLM-generated prose."""

    model_config = ConfigDict(extra="allow")
    schema_version: str = "xai_v1"
    jd_id: str
    cv_id: str
    job_title: str | None = None
    decision: Decision
    dimensions: dict[str, Dimension]
    strength_candidates: list[dict[str, Any]] = []
    gap_candidates: list[dict[str, Any]] = []
    interview_focus: list[dict[str, Any]] = []
    evidence_registry: dict[str, Evidence] = {}

    @model_validator(mode="after")
    def require_dimensions(self) -> "XAIOutput":
        required = {"skill", "experience", "education", "semantic"}
        if set(self.dimensions) != required:
            raise ValueError("dimensions must contain skill, experience, education, semantic")
        for item in self.strength_candidates + self.gap_candidates + self.interview_focus:
            ref = item.get("evidence_ref")
            if ref is not None and ref not in self.evidence_registry:
                raise ValueError(f"unknown evidence_ref: {ref}")
        return self


def compact_llm_payload(xai: XAIOutput) -> dict[str, Any]:
    """Deterministically project the full object into a compact future-LLM input."""
    evidence = {k: v.model_dump(exclude_none=True) for k, v in xai.evidence_registry.items() if k in {i.get("evidence_ref") for i in xai.strength_candidates + xai.gap_candidates + xai.interview_focus}}
    return {"schema_version": "llm_payload_v1", "jd_id": xai.jd_id, "cv_id": xai.cv_id, "job_title": xai.job_title, "decision": xai.decision.model_dump(), "dimensions": {k: v.model_dump() for k, v in xai.dimensions.items()}, "strength_candidates": xai.strength_candidates, "gap_candidates": xai.gap_candidates, "interview_focus": xai.interview_focus, "selected_evidence": evidence}
