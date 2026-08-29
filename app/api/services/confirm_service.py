from __future__ import annotations
import json, re, os
from pathlib import Path
from typing import Any
from app.api.core.config import settings
from app.api.adapters.confirm_override import build_confirm_override, apply_cv_override, apply_jd_override
from app.api.adapters.presentation_adapter import parsed_cv_to_ui_features, parsed_jd_to_ui_features

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")

class ConfirmServiceError(Exception):
    def __init__(self, message: str, status_code: int = 422): super().__init__(message); self.status_code = status_code

def _load(run_id: str, document_id: str, kind: str) -> tuple[dict[str, Any], Path]:
    if not _RUN_ID.fullmatch(run_id): raise ConfirmServiceError("invalid run_id", 400)
    base = Path(settings.RUNTIME_DATA_DIR).parent / ("resumes" if kind == "cv" else "jobs") / run_id / ("resumes" if kind == "cv" else "jobs")
    path = base / ("parsed_resumes.json" if kind == "cv" else "parsed_jds.json")
    if not path.is_file(): raise ConfirmServiceError("runtime parse artifact not found", 404)
    try: records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ConfirmServiceError("runtime parse artifact is malformed", 422) from exc
    if not isinstance(records, list) or not records or not isinstance(records[0], dict) or not isinstance(records[0].get("parsed_data"), dict): raise ConfirmServiceError("runtime parse artifact is malformed", 422)
    parsed = records[0]
    if parsed.get("id") != document_id: raise ConfirmServiceError("document_id does not match runtime artifact", 409)
    return parsed, base

def _atomic(path: Path, data: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

def confirm_document(run_id: str, document_id: str, confirmed_features: list[dict], kind: str) -> dict[str, Any]:
    canonical, base = _load(run_id, document_id, kind)
    original = parsed_cv_to_ui_features(canonical) if kind == "cv" else parsed_jd_to_ui_features(canonical)
    try: override = build_confirm_override(document_id, original, confirmed_features)
    except ValueError as exc: raise ConfirmServiceError(str(exc), 422) from exc
    result = apply_cv_override(canonical, override) if kind == "cv" else apply_jd_override(canonical, override)
    status = "PARTIAL" if result["unsupported_actions"] else "APPLIED"
    _atomic(base / "confirm_override.json", {"schema_version": "confirm_override_v1", **override})
    _atomic(base / ("runtime_parsed_cv.json" if kind == "cv" else "runtime_parsed_jd.json"), result["runtime_document"])
    return {"run_id": run_id, "document_id": document_id, "status": status, "override": override,
            "runtime_parsed": result["runtime_document"], "applied_actions": result["applied_actions"], "unsupported_actions": result["unsupported_actions"]}
