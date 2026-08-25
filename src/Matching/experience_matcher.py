from __future__ import annotations
from typing import Any
from .common import ComponentResult, AVAILABLE, UNKNOWN, NOT_APPLICABLE, aggregate_components, cosine

def match_experience(jd: dict[str, Any], cv: dict[str, Any], cv_artifact: dict[str, Any] | None = None, jd_artifact: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = (config or {}).get("matching", {}).get("experience", config or {})
    yw, ew, threshold = float(cfg.get("years_weight", .3)), float(cfg.get("evidence_weight", .7)), float(cfg.get("semantic_threshold", .6))
    required = jd.get("experience", {}).get("minimum_years")
    candidate = cv.get("experience", {}).get("professional_years")
    if required is None:
        years = ComponentResult(None, NOT_APPLICABLE, "not_required", yw, {"candidate_years": candidate, "minimum_years": None})
    elif required == 0:
        years = ComponentResult(1.0, AVAILABLE, "satisfied", yw, {"candidate_years": candidate, "minimum_years": required})
    elif candidate is None:
        years = ComponentResult(None, UNKNOWN, "unknown_years", yw, {"candidate_years": None, "minimum_years": required})
    else:
        years = ComponentResult(min(float(candidate) / float(required), 1.0), AVAILABLE, "satisfied" if candidate >= required else "below_minimum", yw, {"candidate_years": candidate, "minimum_years": required})
    responsibilities = jd.get("responsibilities", [])
    chunks = (cv_artifact or {}).get("experience_chunks", [])
    evidence_items = []
    for responsibility in responsibilities:
        best = None
        jd_responsibilities = (jd_artifact or {}).get("responsibility_chunks", [])
        normalized_responsibility = str(responsibility).strip().rstrip(".")
        jd_vector = next((item.get("vector") for item in jd_responsibilities if str(item.get("text", "")).strip().rstrip(".") == normalized_responsibility), None)
        for chunk in chunks:
            try: sim = cosine(jd_vector, chunk.get("vector"))
            except (ValueError, TypeError): sim = 0.0
            if best is None or sim > best[0]: best = (sim, chunk)
        if not chunks:
            evidence_items.append({"responsibility": responsibility, "score": None, "status": "source_unavailable"})
        elif best:
            continuous_score = max(0.0, min(1.0, float(best[0])))
            evidence_items.append({"responsibility": responsibility, "score": continuous_score, "status": "matched" if best[0] >= threshold else "low_match", "matched_chunk_id": best[1].get("chunk_id"), "matched_text": best[1].get("text"), "source_type": best[1].get("source_type"), "source_name": best[1].get("source_name"), "raw_similarity": best[0], "threshold": threshold})
        else: evidence_items.append({"responsibility": responsibility, "score": 0.0, "status": "no_evidence"})
    if not responsibilities:
        evidence = ComponentResult(None, NOT_APPLICABLE, "not_required", ew, {"responsibilities": []})
    elif not chunks:
        evidence = ComponentResult(None, UNKNOWN, "source_unavailable", ew, {"responsibilities": evidence_items})
    else:
        vals = [item["score"] for item in evidence_items]
        evidence = ComponentResult(sum(vals) / len(vals), AVAILABLE, "evaluated", ew, {"responsibilities": evidence_items})
    score, coverage = aggregate_components([years, evidence])
    return {"score": score, "coverage": coverage, "years": years.to_dict(), "evidence": evidence.to_dict()}
