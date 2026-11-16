"""Thin bridge to the frozen teammate candidate-feedback generator."""
from __future__ import annotations
from typing import Any, Callable
from LLM_explanation_candidate import (CandidateFeedbackNarrative, CompactPayloadBuilder,
    call_groq_explanation, generate_universal_offline_explanation, get_groq_client_cycler)

def _interview_topics(pre: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"topic": i.get("topic") or "nội dung liên quan",
             "question": f"Vui lòng trình bày kinh nghiệm liên quan đến {i.get('topic') or 'nội dung này'}.",
             "reason": i.get("reason", "deterministic_fact"), "evidence_refs": i.get("evidence_refs", []),
             "fact_id": i.get("fact_id")} for i in pre.get("interview_topics", [])[:3]]

def generate_teammate_narrative(xai: dict[str, Any], pre: dict[str, Any], mode: str = "auto",
                                groq_client_factory: Callable[[], Any] | None = None):
    # Narrative generation still needs a numeric branch for insufficient-data
    # matches; the canonical decision remains untouched in the response.
    payload_xai = xai
    if (xai.get("decision") or {}).get("final_score") is None:
        payload_xai = dict(xai)
        payload_xai["decision"] = dict(xai.get("decision") or {}, final_score=0.0)
    payload = CompactPayloadBuilder.build(payload_xai)
    narrative = None; method = "offline_deterministic"; model = None; fallback = False
    cycler = None
    if mode in {"auto", "groq"}:
        if groq_client_factory is None:
            cycler = get_groq_client_cycler()
        else:
            class _One:
                def __iter__(self): return self
                def __next__(self): return groq_client_factory()
            cycler = _One()
        if mode == "groq" and cycler is None: raise RuntimeError("Groq is unavailable")
        if cycler is not None:
            try:
                narrative = CandidateFeedbackNarrative.model_validate(call_groq_explanation(payload, cycler)).model_dump()
                method, model = "groq_llm", "teammate_candidate_feedback"
            except Exception:
                if mode == "groq": raise
                fallback = True
    if narrative is None: narrative = generate_universal_offline_explanation(payload)
    return {"summary": narrative.get("summary", ""),
            "strengths": [{"text": t, "evidence_refs": []} for t in narrative.get("strengths", [])],
            "gaps": [{"text": t, "evidence_refs": [], "type": "other"} for t in narrative.get("gaps", [])],
            "interview_focus": _interview_topics(pre),
            "disclaimer": "Điểm số và bằng chứng do pipeline MDMS/XAI xác định; LLM chỉ diễn giải các kết quả đã kiểm chứng."}, method, model, fallback
