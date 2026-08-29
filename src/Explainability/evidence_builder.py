"""Deterministic projection of existing matcher/MDMS output into ``xai_v1``."""
from __future__ import annotations

import re
from typing import Any

from .schemas import EvidenceStatus, XAIOutput


def _status(value: Any, *, matched: bool = False) -> EvidenceStatus:
    if matched or str(value).lower() in {"matched", "satisfied", "evaluated"}:
        return EvidenceStatus.MATCHED if matched else EvidenceStatus.AVAILABLE
    value = str(value or "").lower()
    if value in {"unknown", "source_unavailable", "unknown_years"}:
        return EvidenceStatus.UNKNOWN
    if value in {"not_required", "not_applicable"}:
        return EvidenceStatus.NOT_APPLICABLE
    return EvidenceStatus.NO_EVIDENCE


def _safe_id(prefix: str, value: Any) -> str:
    return prefix + re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def build_xai(jd: dict[str, Any], cv: dict[str, Any], matching_result: dict[str, Any], mdms_result: dict[str, Any] | None = None, frozen_weights: dict[str, float] | None = None) -> XAIOutput:
    mdms = mdms_result or matching_result.get("mdms", {})
    # The score and weights must come from one MDMS run.  A supplied frozen
    # vector is only a fallback for legacy artifacts that lack mdms.weights.
    weights = mdms.get("weights") or frozen_weights or {"skill": .4, "experience": .2, "education": .1, "semantic": .3}
    weights = {str(k).removeprefix("w_"): float(v) for k, v in weights.items()}
    effective = {str(k).removeprefix("w_"): float(v) for k, v in mdms.get("effective_weights", weights).items()}
    model_version = mdms.get("model_version") or ("mdms_equal_v1" if all(abs(weights[k] - .25) < 1e-9 for k in weights) else "mdms_tuned_v1" if weights == {"skill": .4, "experience": .2, "education": .1, "semantic": .3} else "mdms_v1")
    registry: dict[str, dict[str, Any]] = {}
    strengths: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    interview: list[dict[str, Any]] = []

    skill = matching_result.get("skill", {})
    skill_reqs = []
    for req in skill.get("requirements", []):
        item = dict(req)
        matched = item.get("status") == "matched"
        if matched and item.get("matched_cv_skill"):
            eid = _safe_id(f"ev_skill_{cv.get('id', 'cv')}_", item.get("jd_skill"))
            registry[eid] = {"evidence_id": eid, "source_type": "cv_skill", "source_id": item.get("matched_cv_skill"), "source_text": item.get("matched_cv_skill"), "matcher": "skill", "score": item.get("score"), "similarity": item.get("similarity")}
            strengths.append({"type": "required_skill_match" if item.get("importance") == "required" else "preferred_skill_match", "dimension": "skill", "requirement": item.get("jd_skill"), "match_type": item.get("match_method"), "evidence_ref": eid})
        elif item.get("importance") == "required" and item.get("status") == "no_evidence":
            gaps.append({"type": "missing_required_skill", "dimension": "skill", "requirement": item.get("jd_skill"), "evidence_ref": None})
            interview.append({"topic": item.get("jd_skill"), "reason": "required_skill_no_evidence", "evidence_ref": None})
        skill_reqs.append(item)
    skill_dim = {"score": skill.get("score"), "status": EvidenceStatus.AVAILABLE if skill.get("score") is not None else _status(skill.get("status")), "weight": effective.get("skill", weights["skill"]), "weighted_contribution": None if skill.get("score") is None else effective.get("skill", weights["skill"]) * skill["score"], "coverage": skill.get("coverage"), "requirements": skill_reqs}

    exp = matching_result.get("experience", {})
    exp_evidence = exp.get("evidence", {})
    exp_items = exp_evidence.get("details", {}).get("responsibilities", [])
    selected = []
    for item in exp_items:
        x = dict(item)
        chunk_id = x.get("matched_chunk_id")
        if chunk_id:
            eid = _safe_id("ev_exp_", chunk_id)
            if eid not in registry:
                registry[eid] = {"evidence_id": eid, "source_type": "cv_" + str(x.get("source_type") or "experience"), "source_id": chunk_id, "source_text": x.get("matched_text"), "matcher": "experience", "score": x.get("score"), "similarity": x.get("raw_similarity")}
            x["evidence_ref"] = eid
            selected.append(x)
            if x.get("status") == "matched": strengths.append({"type": "responsibility_match", "dimension": "experience", "requirement": x.get("responsibility"), "evidence_ref": eid})
            elif x.get("status") in {"low_match", "no_evidence"}:
                interview.append({"topic": x.get("responsibility"), "reason": "weak_responsibility_evidence", "evidence_ref": eid})
    years = exp.get("years", {})
    exp_dim = {"score": exp.get("score"), "status": EvidenceStatus.AVAILABLE if exp.get("score") is not None else _status(exp.get("status")), "weight": effective.get("experience", weights["experience"]), "weighted_contribution": None if exp.get("score") is None else effective.get("experience", weights["experience"]) * exp["score"], "coverage": exp.get("coverage"), "years": years, "evidence": {"score": exp_evidence.get("score"), "status": _status(exp_evidence.get("status")), "responsibilities": selected}}

    edu = matching_result.get("education", {})
    degree, field = edu.get("degree", {}), edu.get("field", {})
    edu_ref = None
    if cv.get("education", {}).get("degree"):
        edu_ref = _safe_id("ev_education_", cv.get("id", "cv"))
        registry[edu_ref] = {"evidence_id": edu_ref, "source_type": "cv_education", "source_id": cv.get("id"), "source_text": f"{cv['education'].get('degree')} — {cv['education'].get('field') or ''}".strip(" —"), "matcher": "education"}
    if degree.get("status") == "satisfied": strengths.append({"type": "degree_requirement_satisfied", "dimension": "education", "requirement": degree.get("details", {}).get("required"), "evidence_ref": edu_ref})
    elif degree.get("status") == "below_minimum": gaps.append({"type": "degree_below_requirement", "dimension": "education", "requirement": degree.get("details", {}).get("required"), "evidence_ref": None})
    edu_dim = {"score": edu.get("score"), "status": EvidenceStatus.AVAILABLE if edu.get("score") is not None else _status(edu.get("status")), "weight": effective.get("education", weights["education"]), "weighted_contribution": None if edu.get("score") is None else effective.get("education", weights["education"]) * edu["score"], "coverage": edu.get("coverage"), "degree": degree, "field": field}

    sem = matching_result.get("semantic", {})
    sem_dim = {"score": sem.get("score"), "status": EvidenceStatus.AVAILABLE if sem.get("score") is not None else _status(sem.get("status")), "weight": effective.get("semantic", weights["semantic"]), "weighted_contribution": None if sem.get("score") is None else effective.get("semantic", weights["semantic"]) * sem["score"], "coverage": sem.get("coverage"), "profile_scope": "global profile-to-role alignment", "raw_similarity": sem.get("raw_similarity")}
    return XAIOutput(schema_version="xai_v1", jd_id=jd.get("id", matching_result.get("jd_id")), cv_id=cv.get("id", matching_result.get("cv_id")), job_title=jd.get("role", {}).get("job_title"), decision={"final_score": mdms.get("final_score"), "status": mdms.get("status", "evaluated"), "coverage": mdms.get("coverage", 0.0), "weights": weights, "effective_weights": effective, "model_version": model_version}, dimensions={"skill": skill_dim, "experience": exp_dim, "education": edu_dim, "semantic": sem_dim}, strength_candidates=strengths, gap_candidates=gaps, interview_focus=interview, evidence_registry=registry)
