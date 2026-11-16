"""Backward-compatible resume parser entry point.

The production implementation lives in ``resume_parser`` and
``offline.resume_offline_parser``.  This thin adapter preserves imports used by
the runtime API without maintaining a second parser implementation.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

from resume_parser import StructuredResume, parse_resume_llm
from offline.resume_offline_parser import OfflineResumeParser

def parse_cleaned_resumes(input_path: Path, output_path: Path, offline: bool = False) -> list[dict[str, Any]]:
    docs = json.loads(Path(input_path).read_text(encoding="utf-8"))
    key = None if offline else os.getenv("GROQ_API_KEY")
    client = Groq(api_key=key) if Groq and key and key.startswith("gsk_") else None
    results = []
    for idx, doc in enumerate(docs, 1):
        text = str(doc.get("content", ""))
        if not text.strip():
            continue
        method = "offline_hybrid"
        structured = None
        if client:
            try:
                structured = parse_resume_llm(text, client)
                method = "groq_llm"
            except Exception:
                structured = None
        if structured is None:
            structured = OfflineResumeParser.parse(text)
        results.append({
            "id": doc.get("id", f"cv_{idx:03d}"),
            "filename": doc.get("filename", ""),
            "domain": doc.get("domain", "IT"),
            "extraction_method": method,
            "source_status": doc.get("status"),
            "parsed_data": structured.model_dump(),
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
