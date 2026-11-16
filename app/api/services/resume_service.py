"""Application orchestration for real CV ingestion (Stage 1 through 3 only)."""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

from app.api.adapters.presentation_adapter import parsed_cv_to_ui_features
from app.api.core.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ResumeParseError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _safe_filename(filename: str) -> str:
    if not filename or not filename.strip():
        raise ResumeParseError("filename is required", 400)
    name = Path(filename).name
    name = _SAFE_NAME.sub("_", name).strip("._")
    if not name:
        raise ResumeParseError("invalid filename", 400)
    return name


def parse_uploaded_cv(filename: str, content: bytes, offline: bool | None = None) -> Dict[str, Any]:
    """Persist one upload, run Stage 1 and Stage 3, and return canonical + UI data."""
    safe_name = _safe_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ResumeParseError(
            "Unsupported media type. Supported formats: PDF, PNG, JPG, JPEG.", 415
        )
    if not content:
        raise ResumeParseError("Uploaded file is empty", 400)
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise ResumeParseError("Uploaded file exceeds the maximum size", 413)

    run_id = uuid.uuid4().hex
    document_id = f"cv_{run_id[:12]}"
    run_dir = Path(settings.RUNTIME_DATA_DIR) / run_id / "resumes"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        raw_path = run_dir / safe_name
        raw_path.write_bytes(content)
    except OSError as exc:
        raise ResumeParseError("Unable to store uploaded file", 500) from exc

    # Data_loader is a script-style module; import it directly without subprocess.
    loader_dir = Path(__file__).resolve().parents[3] / "src" / "Data_loader"
    if str(loader_dir) not in sys.path:
        sys.path.insert(0, str(loader_dir))
    try:
        from document_parser import get_document_parser
        from LLM_parser import parse_cleaned_resumes
    except Exception as exc:
        raise ResumeParseError("CV parser is unavailable", 500) from exc

    try:
        content_text, extraction = get_document_parser().parse(str(raw_path))
    except Exception as exc:
        raise ResumeParseError(f"Document extraction failed: {type(exc).__name__}", 422) from exc
    status = extraction.get("final_status")
    if status == "UNSUPPORTED_FORMAT":
        raise ResumeParseError("Unsupported document format", 415)
    if status == "FAILED":
        detail = extraction.get("error") or "insufficient OCR/document quality"
        raise ResumeParseError(f"Document extraction failed: {detail}", 422)
    if status not in {"ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR", "LOW_QUALITY", "ACCEPTED_DIRECT_TEXT"}:
        raise ResumeParseError("Document extraction failed", 422)

    cleaned = run_dir / "cleaned_resumes.json"
    parsed_out = run_dir / "parsed_resumes.json"
    rel_path = str(raw_path.relative_to(Path.cwd())) if raw_path.is_relative_to(Path.cwd()) else str(raw_path)
    cleaned.write_text(json.dumps([{
        "id": document_id, "filename": filename, "relative_path": rel_path,
        "status": status, "text_length": len(content_text), "content": content_text,
    }], ensure_ascii=False), encoding="utf-8")
    try:
        results = parse_cleaned_resumes(cleaned, parsed_out, offline=settings.OFFLINE_DEFAULT if offline is None else offline)
    except Exception as exc:
        raise ResumeParseError(f"CV structuring failed: {type(exc).__name__}", 422) from exc
    if not results or not isinstance(results[0], dict) or not isinstance(results[0].get("parsed_data"), dict):
        raise ResumeParseError("CV parser returned malformed data", 422)
    parsed = results[0]
    warning = "Document quality is low; structured fields may be incomplete." if status == "LOW_QUALITY" else None
    extraction_meta = {
        "status": status, "text_length": len(content_text),
        "docling_score": extraction.get("docling_score"), "ocr_triggered": bool(extraction.get("ocr_triggered")),
        "ocr_score": extraction.get("ocr_score"), "warning": warning,
    }
    return {"run_id": run_id, "document_id": document_id, "filename": filename,
            "extraction": extraction_meta, "parsed": parsed, "ui_features": parsed_cv_to_ui_features(parsed)}
