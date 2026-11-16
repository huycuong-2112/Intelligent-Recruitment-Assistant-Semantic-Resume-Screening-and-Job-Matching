from __future__ import annotations
from typing import Any

def report_cache_key(match_run_id: str, cv_id: str) -> tuple[str, str]:
    return match_run_id, cv_id

def validate_report_matches_candidate(explanation: dict[str, Any], match_run_id: str, candidate: dict[str, Any], tolerance: float = 1e-6) -> tuple[bool, str | None]:
    if explanation.get("match_run_id") != match_run_id or explanation.get("cv_id") != candidate.get("cv_id", candidate.get("document_id")):
        return False, "Report does not correspond to the selected candidate or match run."
    expected = candidate.get("score_0_1"); actual = (explanation.get("decision") or {}).get("final_score")
    if expected is None or actual is None: return (expected is None and actual is None), None if expected is None and actual is None else "Report score does not correspond to the current match result."
    return (abs(float(expected)-float(actual)) <= tolerance), None if abs(float(expected)-float(actual)) <= tolerance else "Report score does not correspond to the current match result."

def format_report_score(score: Any) -> str:
    return "Insufficient data" if score is None else f"{float(score)*3:.2f} / 3.00"

def report_dimensions(report: dict[str, Any] | None) -> dict[str, Any]:
    """Return canonical 0–1 dimensions without recalculating them."""
    decision = (report or {}).get("decision") or {}
    dimensions = decision.get("dimensions") or {}
    return {name: dimensions.get(name) for name in ("skill", "experience", "education", "semantic")}

def report_weights(report: dict[str, Any] | None) -> dict[str, Any]:
    decision = (report or {}).get("decision") or {}
    return decision.get("effective_weights") or decision.get("weights") or {}

def resolve_report_evidence(report: dict[str, Any] | None, refs: list[str] | None) -> list[dict[str, Any]]:
    """Resolve optional server-provided evidence while retaining IDs privately."""
    report = report or {}; refs = refs or []
    registry = report.get("evidence_registry") or report.get("selected_evidence") or {}
    resolved=[]
    for ref in refs:
        item = registry.get(ref) if isinstance(registry, dict) else None
        if isinstance(item, dict):
            resolved.append({"ref": ref, **item})
    return resolved
