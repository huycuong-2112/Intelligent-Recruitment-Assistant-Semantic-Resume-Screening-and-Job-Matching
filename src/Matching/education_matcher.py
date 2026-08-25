from __future__ import annotations
from typing import Any
from .common import ComponentResult, AVAILABLE, UNKNOWN, NOT_APPLICABLE, aggregate_components

DEGREE_ORDER = {"High School": 0, "Associate": 1, "Bachelor": 2, "Engineer": 2, "Master": 3, "Ph.D": 4, "Any": -1}

def match_education(jd: dict[str, Any], cv: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = (config or {}).get("matching", {}).get("education", config or {})
    dw, fw = float(cfg.get("degree_weight", .6)), float(cfg.get("field_weight", .4))
    required = jd.get("education", {}).get("minimum_degree"); candidate = cv.get("education", {}).get("degree")
    if not required: degree = ComponentResult(None, NOT_APPLICABLE, "not_required", dw, {"candidate": candidate, "required": required})
    elif candidate is None:
        source_unavailable = str(cv.get("source_status", "")).upper() in {"FAILED", "UNAVAILABLE", "EXTRACTION_FAILED"}
        evaluable = bool(cv.get("skills", {}).get("all") or cv.get("experience", {}).get("work_evidence") or cv.get("projects"))
        degree = ComponentResult(None if source_unavailable or not evaluable else 0.0, UNKNOWN if source_unavailable or not evaluable else AVAILABLE, "source_unavailable" if source_unavailable or not evaluable else "no_evidence", dw, {"candidate": None, "required": required})
    else: degree = ComponentResult(1.0 if DEGREE_ORDER.get(candidate, -1) >= DEGREE_ORDER.get(required, 99) else 0.0, AVAILABLE, "satisfied" if DEGREE_ORDER.get(candidate, -1) >= DEGREE_ORDER.get(required, 99) else "below_minimum", dw, {"candidate": candidate, "required": required})
    fields = jd.get("education", {}).get("preferred_fields", []) or []
    candidate_field = cv.get("education", {}).get("field")
    if not fields: field = ComponentResult(None, NOT_APPLICABLE, "not_required", fw, {"candidate": candidate_field, "preferred_fields": fields})
    elif candidate_field is None:
        source_unavailable = str(cv.get("source_status", "")).upper() in {"FAILED", "UNAVAILABLE", "EXTRACTION_FAILED"}
        evaluable = bool(cv.get("skills", {}).get("all") or cv.get("experience", {}).get("work_evidence") or cv.get("projects"))
        field = ComponentResult(None if source_unavailable or not evaluable else 0.0, UNKNOWN if source_unavailable or not evaluable else AVAILABLE, "source_unavailable" if source_unavailable or not evaluable else "no_evidence", fw, {"candidate": None, "preferred_fields": fields})
    else:
        matched = any(str(candidate_field).casefold() == str(value).casefold() for value in fields)
        field = ComponentResult(1.0 if matched else 0.0, AVAILABLE, "matched" if matched else "no_evidence", fw, {"candidate": candidate_field, "preferred_fields": fields})
    score, coverage = aggregate_components([degree, field])
    return {"score": score, "coverage": coverage, "degree": degree.to_dict(), "field": field.to_dict()}
