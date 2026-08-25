from __future__ import annotations
from typing import Any
from .common import ComponentResult, AVAILABLE, UNKNOWN, cosine

def _compatible(cv_artifact: dict[str, Any], jd_artifact: dict[str, Any]) -> None:
    cv_model, jd_model = cv_artifact.get("model", {}), jd_artifact.get("model", {})
    if cv_model.get("name") != jd_model.get("name") or cv_model.get("dimension") != jd_model.get("dimension"):
        raise ValueError("incompatible embedding model metadata")

def match_semantic(cv_artifact: dict[str, Any] | None, jd_artifact: dict[str, Any] | None) -> dict[str, Any]:
    if not cv_artifact or not jd_artifact or not cv_artifact.get("profile", {}).get("vector") or not jd_artifact.get("profile", {}).get("vector"):
        return {"score": None, "coverage": 0.0, "status": "source_unavailable", "availability": UNKNOWN}
    _compatible(cv_artifact, jd_artifact)
    raw = cosine(cv_artifact["profile"]["vector"], jd_artifact["profile"]["vector"])
    return {"score": max(0.0, min(1.0, raw)), "coverage": 1.0, "status": "evaluated", "availability": AVAILABLE, "raw_similarity": raw}
