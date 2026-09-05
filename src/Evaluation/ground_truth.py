from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class CandidateAnnotation:
    cv_id: str; relevance_grade: int; overall: int; annotator: str | None = None; notes: str | None = None; split: str | None = None; annotator_1: int | None = None; annotator_2: int | None = None; annotator_3: int | None = None

@dataclass(frozen=True)
class GroundTruthRecord:
    jd_id: str; domain: str; candidates: tuple[CandidateAnnotation, ...]; annotation_metadata: dict[str, Any]

def validate_ground_truth(data: dict[str, Any]) -> GroundTruthRecord:
    if not isinstance(data, dict) or not isinstance(data.get("jd_id"), str) or not data["jd_id"]: raise ValueError("jd_id is required")
    if not isinstance(data.get("domain"), str) or not data["domain"]: raise ValueError("domain is required")
    raw = data.get("candidates")
    if not isinstance(raw, list): raise ValueError("candidates must be a list")
    seen: set[str] = set(); candidates = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("cv_id"), str) or not item["cv_id"]: raise ValueError("candidate cv_id is required")
        if item["cv_id"] in seen: raise ValueError(f"duplicate cv_id: {item['cv_id']}")
        seen.add(item["cv_id"])
        grade = item.get("overall", item.get("relevance_grade")); score = item.get("ground_truth_score")
        if not isinstance(grade, int) or grade not in {0, 1, 2, 3}: raise ValueError("relevance_grade must be one of 0,1,2,3")
        if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 100): raise ValueError("ground_truth_score must be between 0 and 100")
        ratings = [item.get(f"annotator_{i}") for i in range(1,4)]
        if any(value is not None and (not isinstance(value, int) or value not in {0,1,2,3}) for value in ratings): raise ValueError("annotator ratings must be integers in 0..3")
        candidates.append(CandidateAnnotation(item["cv_id"], grade, grade, item.get("annotator"), item.get("notes"), item.get("split"), *ratings))
    return GroundTruthRecord(data["jd_id"], data["domain"], tuple(candidates), data.get("annotation_metadata", {}))

def load_ground_truth(path: str) -> GroundTruthRecord:
    import json
    from pathlib import Path
    file_path = Path(path)
    if not file_path.exists(): raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    return validate_ground_truth(json.loads(file_path.read_text(encoding="utf-8")))
