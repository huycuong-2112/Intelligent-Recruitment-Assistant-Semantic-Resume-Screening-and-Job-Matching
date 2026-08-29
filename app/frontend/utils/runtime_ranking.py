"""Deterministic ranking projection over an existing runtime match response."""
from __future__ import annotations
from typing import Any

def build_ranking_rows(response: dict[str, Any], filename_by_document_id: dict[str, str]) -> list[dict[str, Any]]:
    rows=[]
    for index, item in enumerate(response.get("results", [])):
        components=item.get("components", {}) or {}
        scores={k:(components.get(k) or {}).get("score") for k in ("skill","experience","education","semantic")}
        rows.append({"rank":None,"filename":filename_by_document_id.get(item.get("cv_id"), item.get("cv_id")),"document_id":item.get("cv_id"),"score_0_1":item.get("score_0_1"),"score_0_3":item.get("score_0_3"),"status":item.get("status"),"coverage":item.get("coverage"),"skill_score":scores["skill"],"experience_score":scores["experience"],"education_score":scores["education"],"semantic_score":scores["semantic"],"components":components,"mdms":item.get("mdms"),"_order":index})
    rows.sort(key=lambda r:(r["score_0_1"] is not None, r["score_0_1"] if r["score_0_1"] is not None else -1, -r["_order"]), reverse=True)
    for i,row in enumerate(rows,1): row["rank"] = i if row["score_0_1"] is not None else None
    return rows
