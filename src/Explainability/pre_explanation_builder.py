"""Compact, deterministic projection from canonical xai_v1 to pre_explanation_v1."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _status(value: Any) -> str:
    return str(value or "").upper()


def _refs(item: dict[str, Any]) -> list[str]:
    value = item.get("evidence_refs", item.get("evidence_ref", []))
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _assert_no_vectors(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "vector" in str(key).lower() or "embedding" in str(key).lower():
                raise ValueError("pre-explanation cannot contain vectors or embeddings")
            _assert_no_vectors(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_vectors(child)


def build_pre_explanation(xai: dict[str, Any]) -> dict[str, Any]:
    if xai.get("schema_version") != "xai_v1":
        raise ValueError("expected source schema_version xai_v1")
    for field in ("cv_id", "jd_id", "decision", "dimensions"):
        if not xai.get(field):
            raise ValueError(f"missing required XAI field: {field}")
    dimensions = xai["dimensions"]
    required_dimensions = {"skill", "experience", "education", "semantic"}
    if set(dimensions) != required_dimensions:
        raise ValueError("XAI must contain all four dimensions")

    skill_reqs = dimensions["skill"].get("requirements", [])
    required_missing = [r["jd_skill"] for r in skill_reqs if str(r.get("importance", "")).lower() == "required" and _status(r.get("status")) == "NO_EVIDENCE"]
    strengths = []
    for item in xai.get("strength_candidates", [])[:3]:
        fact = {"degree_requirement_satisfied": "Degree requirement satisfied.", "required_skill_match": "Required skill match found.", "preferred_skill_match": "Preferred skill match found.", "responsibility_match": "Responsibility evidence matched."}.get(item.get("type"), "Deterministic matcher evidence recorded.")
        strengths.append({"type": item.get("type"), "fact": fact, "value": item.get("requirement") or item.get("value"), "evidence_refs": _refs(item)})

    weak = []
    responsibilities = dimensions["experience"].get("evidence", {}).get("responsibilities", [])
    for item in responsibilities:
        if _status(item.get("status")) == "LOW_MATCH" and len(weak) < 2:
            weak.append({"requirement": item.get("responsibility"), "status": "LOW_MATCH", "evidence_text": item.get("matched_text"), "evidence_refs": _refs(item)})

    unknowns = []
    for name, dim in dimensions.items():
        if _status(dim.get("status")) == "UNKNOWN":
            unknowns.append({"dimension": name, "status": "UNKNOWN"})
        for subname in ("years", "degree", "field", "evidence"):
            sub = dim.get(subname)
            if isinstance(sub, dict) and _status(sub.get("status")) == "UNKNOWN":
                unknowns.append({"dimension": name, "component": subname, "status": "UNKNOWN"})

    canonical_topics = [{"topic": item.get("topic"), "reason": item.get("reason"), "evidence_refs": _refs(item)} for item in xai.get("interview_focus", [])]
    topics = canonical_topics[:3]
    # Preserve canonical priority while ensuring a mixed-dimension signal when
    # both required gaps and weak responsibility evidence exist.
    if required_missing and weak and not any(t.get("reason") == "weak_responsibility_evidence" for t in topics):
        topics = [t for t in topics if t.get("reason") != "weak_responsibility_evidence"][:2]
        topics.append({"topic": weak[0]["requirement"], "reason": "weak_experience_evidence", "evidence_refs": weak[0]["evidence_refs"]})
    if not topics:
        topics = [{"topic": s, "reason": "required_skill_no_evidence", "evidence_refs": []} for s in required_missing[:3]]
        topics.extend({"topic": item["requirement"], "reason": "weak_experience_evidence", "evidence_refs": item["evidence_refs"]} for item in weak[: max(0, 3 - len(topics))])

    refs: list[str] = []
    for item in strengths + weak + topics:
        for ref in item.get("evidence_refs", []):
            if ref not in refs:
                refs.append(ref)
    registry = xai.get("evidence_registry", {})
    selected = {}
    for ref in refs:
        if ref not in registry:
            raise ValueError(f"unresolved evidence reference: {ref}")
        source = registry[ref]
        selected[ref] = {k: source[k] for k in ("source_type", "source_name", "source_id", "source_text") if k in source}

    decision = xai["decision"]
    education_details = None
    if "degree" in dimensions["education"] or "field" in dimensions["education"]:
        education_details = {"degree": {"status": _status(dimensions["education"].get("degree", {}).get("status")), "candidate_value": dimensions["education"].get("degree", {}).get("details", {}).get("candidate"), "required_value": dimensions["education"].get("degree", {}).get("details", {}).get("required")}, "field": {"status": _status(dimensions["education"].get("field", {}).get("status")), "candidate_value": dimensions["education"].get("field", {}).get("details", {}).get("candidate"), "preferred_fields": dimensions["education"].get("field", {}).get("details", {}).get("preferred_fields", [])}}
    for i, item in enumerate(strengths, 1): item.setdefault("fact_id", f"str_{i:03d}")
    for i, item in enumerate(weak, 1): item.setdefault("fact_id", f"gap_{i:03d}")
    for i, item in enumerate(topics, 1): item.setdefault("fact_id", f"int_{i:03d}")
    facts = {"strengths": strengths, "required_skills_no_evidence": required_missing, "weak_experience_evidence": weak, "unknowns": unknowns}
    if education_details is not None:
        facts["education_details"] = education_details
    output = {"schema_version": "pre_explanation_v1", "source_schema_version": "xai_v1", "cv_id": xai["cv_id"], "jd_id": xai["jd_id"], "target_role": xai.get("job_title"), "model_version": decision.get("model_version"), "decision": {"final_score": decision.get("final_score"), "coverage": decision.get("coverage"), "weights": decision.get("weights"), "dimensions": {name: dimensions[name].get("score") for name in required_dimensions}}, "facts": facts, "interview_topics": topics[:3], "interview_config": {"max_questions": 3}, "selected_evidence": selected, "rendering_rules": {"language": "vi", "max_strengths": 3, "max_gaps": 3, "max_interview_questions": 3, "no_evidence_means_absence_of_documented_evidence": True, "unknown_is_not_negative": True, "semantic_cannot_prove_specific_skills": True}}
    _assert_no_vectors(output)
    return output
