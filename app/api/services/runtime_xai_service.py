"""Server-authoritative bridge from persisted runtime matches to xai_v1."""
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any

from app.api.core.config import settings
from src.Explainability.evidence_builder import build_xai
from src.Explainability.schemas import XAIOutput

_HEX = re.compile(r"^[0-9a-f]{32}$")
_CV = re.compile(r"^cv_[A-Za-z0-9_.-]+$")

class RuntimeXAIError(ValueError):
    pass

def _safe(value: str, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise RuntimeXAIError(f"invalid {label}")

def _load_json(path: Path, label: str) -> dict[str, Any]:
    try: value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise RuntimeXAIError(f"{label} is unavailable or malformed") from exc
    if not isinstance(value, dict): raise RuntimeXAIError(f"{label} is malformed")
    return value

def _prepared(run_id: str, document_id: str, kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base=Path(settings.RUNTIME_DATA_DIR).parent/("resumes" if kind=="cv" else "jobs")/run_id/("resumes" if kind=="cv" else "jobs")/"prepared"
    name="normalized_cv.json" if kind=="cv" else "normalized_jd.json"
    normalized=_load_json(base/name, f"prepared {kind} normalization")
    manifest=_load_json(base/"preparation_manifest.json", f"prepared {kind} manifest")
    if normalized.get("id") != document_id: raise RuntimeXAIError(f"prepared {kind} identity mismatch")
    return normalized, manifest

def build_runtime_xai(match_run_id: str, cv_id: str) -> dict[str, Any]:
    _safe(match_run_id, _HEX, "match_run_id"); _safe(cv_id, _CV, "cv_id")
    root=Path(settings.RUNTIME_DATA_DIR).parent/"matching"/match_run_id
    manifest=_load_json(root/"match_manifest.json", "match manifest")
    if manifest.get("match_run_id") != match_run_id: raise RuntimeXAIError("match manifest identity mismatch")
    candidate_refs={x.get("document_id"):x for x in manifest.get("candidates",[]) if isinstance(x,dict)}
    ref=candidate_refs.get(cv_id)
    if not ref: raise RuntimeXAIError("candidate does not belong to match run")
    candidate=_load_json(root/"candidates"/f"{cv_id}.json", "candidate match result")
    if candidate.get("cv_id") != cv_id or candidate.get("jd_id") != manifest.get("job",{}).get("document_id"): raise RuntimeXAIError("candidate match identity mismatch")
    jd_ref=manifest.get("job",{}); jd_id=jd_ref.get("document_id")
    if not isinstance(jd_id,str) or not jd_id.startswith("jd_"): raise RuntimeXAIError("invalid job identity in manifest")
    jd, jd_manifest=_prepared(jd_ref.get("run_id"), jd_id, "jd")
    cv, cv_manifest=_prepared(ref.get("run_id"), cv_id, "cv")
    if jd_manifest.get("runtime_input_sha256") != jd_ref.get("runtime_input_sha256") or cv_manifest.get("runtime_input_sha256") != ref.get("runtime_input_sha256"):
        raise RuntimeXAIError("Prepared runtime artifacts no longer correspond to this match run. Re-run Matching before generating XAI.")
    mdms=dict(candidate.get("mdms") or {})
    mdms["weights"]=mdms.get("runtime_weights") or manifest.get("mdms_config",{}).get("weights")
    mdms["model_version"]=(mdms.get("weights_metadata") or {}).get("version") or manifest.get("mdms_config",{}).get("version")
    matching={**candidate.get("components",{}), "mdms":mdms, "jd_id":jd_id, "cv_id":cv_id}
    xai=XAIOutput.model_validate(build_xai(jd, cv, matching))
    output=xai.model_dump(mode="json")
    output.update({"match_run_id":match_run_id,"runtime_input_sha256":{"jd":jd_ref.get("runtime_input_sha256"),"cv":ref.get("runtime_input_sha256")},"mdms_config_version":mdms.get("model_version")})
    target=root/"xai"; target.mkdir(parents=True,exist_ok=True); path=target/f"{cv_id}.json"; tmp=path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,path)
    return output
