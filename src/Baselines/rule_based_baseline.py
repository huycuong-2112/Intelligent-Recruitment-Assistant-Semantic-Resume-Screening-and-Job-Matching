from __future__ import annotations
from typing import Any
from src.Normalization.skill_normalizer import normalize_skill
from src.Matching.education_matcher import match_education

def run_rule_based(jd: dict[str, Any], cv: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = (config or {}).get("baseline", {}).get("rule_based", {}); sw, ew, dw = float(cfg.get("skill_weight", .5)), float(cfg.get("experience_weight", .3)), float(cfg.get("education_weight", .2))
    required, preferred = jd.get("skills", {}).get("required", []), jd.get("skills", {}).get("preferred", []); cv_skills = {normalize_skill(x) for x in cv.get("skills", {}).get("all", [])}
    skill_den = len(required) + .5 * len(preferred); skill_score = (sum(normalize_skill(x) in cv_skills for x in required) + .5 * sum(normalize_skill(x) in cv_skills for x in preferred)) / skill_den if skill_den else None
    minimum, years = jd.get("experience", {}).get("minimum_years"), cv.get("experience", {}).get("professional_years")
    if minimum is None: exp_score = None
    elif minimum == 0: exp_score = 1.0
    elif years is None: exp_score = None
    else: exp_score = min(float(years) / float(minimum), 1.0)
    education = match_education(jd, cv); edu_score = education.get("score")
    components = [score for score in (skill_score, exp_score, edu_score) if score is not None]; score = sum(weight * value for weight, value in ((sw, skill_score), (ew, exp_score), (dw, edu_score)) if value is not None) / sum(weight for weight, value in ((sw, skill_score), (ew, exp_score), (dw, edu_score)) if value is not None) if components else None
    coverage = sum(weight for weight, value in ((sw, skill_score), (ew, exp_score), (dw, edu_score)) if value is not None) / (sw + ew + dw)
    return {"method": "rule_based", "jd_id": jd.get("id"), "cv_id": cv.get("id"), "score_0_1": score, "score_0_100": score * 100 if score is not None else None, "components": {"skill": skill_score, "experience": exp_score, "education": edu_score}, "coverage": coverage, "status": "evaluated" if score is not None else "insufficient_data", "weights": {"skill": sw, "experience": ew, "education": dw, "weight_policy": "HEURISTIC_NOT_OPTIMIZED"}}
