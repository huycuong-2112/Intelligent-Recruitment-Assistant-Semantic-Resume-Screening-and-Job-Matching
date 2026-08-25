"""Structure-aware semantic evidence chunking for normalized CVs and JDs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .feature_builder import CVFeatures, JDFeatures, build_cv_features, build_jd_features


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    document_id: str
    source_type: str
    text: str
    source_name: str | None = None
    metadata: dict[str, Any] | None = None


def _sentences(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=(?:and|then|also)\s+(?:built|developed|designed|implemented|engineered|created|deployed)\b)", text, flags=re.IGNORECASE)
    cleaned = []
    for part in parts:
        part = re.sub(r"^(?:and|then|also)\s+", "", part, flags=re.IGNORECASE).strip(" .")
        if len(part.split()) >= 2 and part not in cleaned:
            cleaned.append(part + ("." if not part.endswith((".", "!", "?")) else ""))
    return cleaned


def _record_text(item: Any) -> tuple[str, str | None, dict[str, Any]]:
    if isinstance(item, str):
        return item, None, {}
    if isinstance(item, dict):
        name = item.get("name") or item.get("title") or item.get("project_name") or item.get("company")
        text = item.get("description") or item.get("summary") or item.get("details") or item.get("achievement") or item.get("impact")
        if text is None:
            text = item.get("responsibilities_and_impact")
        metrics = item.get("achievements") or item.get("impact_metrics")
        if metrics and text:
            values = metrics if isinstance(metrics, list) else [metrics]
            text = f"{text} Impact: {'; '.join(str(v) for v in values)}"
        if isinstance(text, list):
            text = " ".join(str(value).strip() for value in text if str(value).strip())
        return str(text or ""), str(name) if name else None, {"raw": item, "company": item.get("company"), "role": item.get("role")}
    return "", None, {}


def build_cv_chunks(value: CVFeatures | dict[str, Any]) -> list[EvidenceChunk]:
    features = value if isinstance(value, CVFeatures) else build_cv_features(value)
    document_id = features.id or "cv_unknown"
    chunks: list[EvidenceChunk] = []
    counters = {"work": 0, "project": 0}
    for source_type, entries in (("work", features.work_evidence), ("project", features.projects or features.project_evidence)):
        for entry in entries:
            text, name, metadata = _record_text(entry)
            for sentence in _sentences(text):
                counters[source_type] += 1
                chunks.append(EvidenceChunk(f"{document_id}_{source_type}_{counters[source_type]:03d}", document_id, source_type, sentence, name, metadata))
    return chunks


def build_jd_chunks(value: JDFeatures | dict[str, Any]) -> list[EvidenceChunk]:
    features = value if isinstance(value, JDFeatures) else build_jd_features(value)
    document_id = features.id or "jd_unknown"
    chunks: list[EvidenceChunk] = []
    index = 0
    overview = features.role.get("overview")
    for source_type, entries in (("overview", [overview]), ("responsibility", features.responsibilities)):
        for entry in entries:
            for sentence in _sentences(entry):
                index += 1
                chunks.append(EvidenceChunk(f"{document_id}_{source_type}_{index:03d}", document_id, source_type, sentence))
    # Keep IDs deterministic even when overview is absent and responsibilities exist.
    return [EvidenceChunk(c.chunk_id.replace(f"_{i:03d}", f"_{i:03d}"), c.document_id, c.source_type, c.text, c.source_name, c.metadata) for i, c in enumerate(chunks, 1)]


def chunks_for_debug(chunks: Iterable[EvidenceChunk]) -> list[dict[str, Any]]:
    return [asdict(chunk) for chunk in chunks]


def format_chunk_debug(chunks: Iterable[EvidenceChunk]) -> str:
    return "\n".join(f"{c.document_id} | {c.chunk_id} | {c.source_type} | {c.source_name or '-'} | {c.text}" for c in chunks)
