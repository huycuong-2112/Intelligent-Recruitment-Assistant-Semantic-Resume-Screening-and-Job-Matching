from __future__ import annotations
from typing import Any
from .common import ComponentResult, AVAILABLE, UNKNOWN, aggregate_components, cosine
from src.Normalization.skill_normalizer import normalize_skill
from src.Normalization.taxonomy import RELATED_CONCEPTS

def _provenance(skills: list[dict[str, Any]], name: str) -> list[str]:
    for item in skills:
        if item.get("skill") == name:
            source = item.get("source", [])
            return source if isinstance(source, list) else [source]
    return []

def match_skills(jd: dict[str, Any], cv: dict[str, Any], cv_artifact: dict[str, Any] | None = None, config: dict[str, Any] | None = None, jd_artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = (config or {}).get("matching", {}).get("skills", config or {})
    req_weight, pref_weight = float(cfg.get("required_weight", 1.0)), float(cfg.get("preferred_weight", 0.5))
    related_score, threshold = float(cfg.get("taxonomy_related_score", 0.7)), float(cfg.get("semantic_threshold", 0.75))
    cv_skills = cv.get("skills", {}).get("all", [])
    artifact_skills = (cv_artifact or {}).get("skills", [])
    by_skill = {item.get("skill"): item for item in artifact_skills if isinstance(item, dict)}
    requirements, components, matched_required, matched_preferred, missing_required, missing_preferred = [], [], [], [], [], []
    related_groups = [set(values) for values in RELATED_CONCEPTS.values()]
    for importance, values, weight in (("required", jd.get("skills", {}).get("required", []), req_weight), ("preferred", jd.get("skills", {}).get("preferred", []), pref_weight)):
        for skill in values:
            canonical = normalize_skill(skill); score = 0.0; method = None; matched = None; similarity = None; status = "no_evidence"
            for candidate in cv_skills:
                if normalize_skill(candidate) == canonical: score, method, matched, status = 1.0, "exact", candidate, "matched"; break
            if matched is None and any(canonical in group and any(normalize_skill(x) in group for x in cv_skills) for group in related_groups):
                score, method, status = related_score, "taxonomy_related", "matched"
            if matched is None and by_skill and cv_artifact:
                jd_items = (jd_artifact or {}).get("required_skills", []) + (jd_artifact or {}).get("preferred_skills", [])
                jd_item = next((x for x in jd_items if x.get("skill") == canonical), None)
                if jd_item:
                    candidates = [(name, cosine(jd_item.get("vector"), item.get("vector"))) for name, item in by_skill.items()]
                    if candidates:
                        matched, similarity = max(candidates, key=lambda x: x[1])
                        if similarity >= threshold: score, method, status = similarity, "semantic", "matched"
                        else: matched = None; similarity = similarity
            detail = {"jd_skill": canonical, "importance": importance, "score": score, "availability": AVAILABLE, "status": status, "match_method": method, "matched_cv_skill": matched, "cv_skill_provenance": _provenance(artifact_skills, matched) if matched else [], "similarity": similarity, "weight": weight}
            requirements.append(detail); components.append(ComponentResult(score, AVAILABLE, status, weight, detail))
            (matched_required if importance == "required" and status == "matched" else matched_preferred if status == "matched" else missing_required if importance == "required" else missing_preferred).append(canonical)
    score, coverage = aggregate_components(components)
    if not components: return {"score": None, "coverage": 0.0, "requirements": [], "status": "not_required"}
    return {"score": score, "coverage": coverage, "matched_required": matched_required, "matched_preferred": matched_preferred, "missing_required": missing_required, "missing_preferred": missing_preferred, "requirements": requirements}
