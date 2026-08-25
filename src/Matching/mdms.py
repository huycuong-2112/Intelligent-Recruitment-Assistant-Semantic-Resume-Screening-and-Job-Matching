from __future__ import annotations
from typing import Any
from .common import NOT_APPLICABLE, UNKNOWN

def aggregate_mdms(results: dict[str, dict[str, Any]], weights: dict[str, float]) -> dict[str, Any]:
    if set(weights) != {"skill", "experience", "education", "semantic"}: raise ValueError("weights must cover skill, experience, education, semantic")
    if any(value < 0 for value in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6: raise ValueError("weights must be non-negative and sum to 1")
    applicable = {name: result for name, result in results.items() if result.get("availability") != NOT_APPLICABLE and result.get("status") != "not_required"}
    unknown = [name for name, result in applicable.items() if result.get("score") is None or result.get("availability") == UNKNOWN]
    if unknown: return {"final_score": None, "status": "insufficient_data", "coverage": sum(weights[n] * float(applicable[n].get("coverage", 0.0)) for n in applicable) / sum(weights[n] for n in applicable) if applicable else 0.0, "effective_weights": {n: weights[n] / sum(weights[x] for x in applicable) for n in applicable}}
    denominator = sum(weights[n] for n in applicable)
    effective = {n: weights[n] / denominator for n in applicable}
    score = sum(effective[n] * float(applicable[n]["score"]) for n in applicable)
    coverage = sum(effective[n] * float(applicable[n].get("coverage", 0.0)) for n in applicable)
    return {"final_score": score, "status": "evaluated", "coverage": coverage, "effective_weights": effective}
