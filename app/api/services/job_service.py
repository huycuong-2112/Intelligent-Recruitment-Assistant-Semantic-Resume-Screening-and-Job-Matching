from __future__ import annotations
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict
from app.api.adapters.presentation_adapter import parsed_jd_to_ui_features
from app.api.core.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

class JobParseError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message); self.status_code = status_code

def _safe_filename(filename: str) -> str:
    if not filename or not filename.strip(): raise JobParseError("filename is required", 400)
    name = _SAFE_NAME.sub("_", Path(filename).name).strip("._")
    if not name: raise JobParseError("invalid filename", 400)
    return name

def parse_uploaded_job(filename: str, content: bytes, offline: bool | None = None) -> Dict[str, Any]:
    safe_name = _safe_filename(filename)
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise JobParseError("Unsupported media type. Supported formats: PDF, PNG, JPG, JPEG.", 415)
    if not content: raise JobParseError("Uploaded file is empty", 400)
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES: raise JobParseError("Uploaded file exceeds the maximum size", 413)
    run_id = uuid.uuid4().hex; document_id = f"jd_{run_id[:12]}"
    run_dir = Path(settings.RUNTIME_DATA_DIR).parent / "jobs" / run_id / "jobs"
    try:
        run_dir.mkdir(parents=True, exist_ok=False); raw_path = run_dir / safe_name; raw_path.write_bytes(content)
    except OSError as exc: raise JobParseError("Unable to store uploaded file", 500) from exc
    loader_dir = Path(__file__).resolve().parents[3] / "src" / "Data_loader"
    if str(loader_dir) not in sys.path: sys.path.insert(0, str(loader_dir))
    try:
        from document_parser import get_document_parser
        from jd_parser import parse_cleaned_jds
        text, extraction = get_document_parser().parse(str(raw_path))
    except Exception as exc: raise JobParseError("Document extraction failed", 422) from exc
    status = extraction.get("final_status")
    if status == "UNSUPPORTED_FORMAT": raise JobParseError("Unsupported document format", 415)
    if status == "FAILED" or status not in {"ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR", "LOW_QUALITY", "ACCEPTED_DIRECT_TEXT"}:
        raise JobParseError("Document extraction failed", 422)
    cleaned = run_dir / "cleaned_jds.json"; parsed_out = run_dir / "parsed_jds.json"
    rel = str(raw_path)
    cleaned.write_text(json.dumps([{"id": document_id, "filename": filename, "relative_path": rel, "status": status, "text_length": len(text), "content": text}], ensure_ascii=False), encoding="utf-8")
    try: results = parse_cleaned_jds(cleaned, parsed_out, offline=settings.OFFLINE_DEFAULT if offline is None else offline)
    except Exception as exc: raise JobParseError("JD structuring failed", 422) from exc
    if not results or not isinstance(results[0], dict) or not isinstance(results[0].get("parsed_data"), dict): raise JobParseError("JD parser returned malformed data", 422)
    parsed = results[0]; warning = "Document quality is low; structured fields may be incomplete." if status == "LOW_QUALITY" else None
    extraction_meta = {"status": status, "text_length": len(text), "docling_score": extraction.get("docling_score"), "ocr_triggered": bool(extraction.get("ocr_triggered")), "ocr_score": extraction.get("ocr_score"), "warning": warning}
    return {"run_id": run_id, "document_id": document_id, "filename": filename, "extraction": extraction_meta, "parsed": parsed, "ui_features": parsed_jd_to_ui_features(parsed)}
