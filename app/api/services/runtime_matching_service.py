"""Structured runtime matching orchestration for confirmed CV/JD artifacts."""
from __future__ import annotations

import json, os, uuid
from pathlib import Path
from typing import Any

from app.api.core.config import settings
from app.api.services.runtime_matching_config import load_runtime_matching_config
from app.api.services.runtime_preparation_service import prepare_cv_runtime, prepare_jd_runtime, RuntimePreparationError
from src.Matching.skill_matcher import match_skills
from src.Matching.experience_matcher import match_experience
from src.Matching.education_matcher import match_education
from src.Matching.semantic_matcher import match_semantic
from src.Matching.mdms import aggregate_mdms
from src.Representation.embedding_service import EmbeddingService

class RuntimeMatchingError(ValueError):
    pass

def _ref(ref: Any, prefix: str) -> tuple[str, str]:
    if not isinstance(ref, dict): raise RuntimeMatchingError("document reference must be an object")
    run_id, doc_id = ref.get("run_id"), ref.get("document_id")
    if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id): raise RuntimeMatchingError("invalid run_id")
    if not isinstance(doc_id, str) or not doc_id.startswith(prefix) or not doc_id.strip(): raise RuntimeMatchingError(f"invalid {prefix} document_id")
    return run_id, doc_id

def run_runtime_matching(domain: str, job_ref: dict[str, str], candidate_refs: list[dict[str, str]], embedding_service: Any = None) -> dict[str, Any]:
    if not isinstance(candidate_refs, list) or not candidate_refs: raise RuntimeMatchingError("at least one candidate is required")
    try:
        config = load_runtime_matching_config(domain)
    except Exception as exc:
        raise RuntimeMatchingError(str(exc)) from exc
    jrun, jid = _ref(job_ref, "jd_")
    refs = [_ref(ref, "cv_") for ref in candidate_refs]
    if len({doc for _, doc in refs}) != len(refs): raise RuntimeMatchingError("duplicate candidate document_id")
    try:
        service = embedding_service or EmbeddingService()
        jd = prepare_jd_runtime(jrun, jid, domain, service)
        results = []
        for crun, cid in refs:
            cv = prepare_cv_runtime(crun, cid, domain, service)
            components = {
                "skill": match_skills(jd["normalized"], cv["normalized"], cv["embedding_artifact"], _config_for_matchers(config), jd["embedding_artifact"]),
                "experience": match_experience(jd["normalized"], cv["normalized"], cv["embedding_artifact"], jd["embedding_artifact"], _config_for_matchers(config)),
                "education": match_education(jd["normalized"], cv["normalized"], _config_for_matchers(config)),
                "semantic": match_semantic(cv["embedding_artifact"], jd["embedding_artifact"]),
            }
            mdms = aggregate_mdms(components, config["weights"])
            score = mdms.get("final_score")
            result = {"jd_id": jid, "cv_id": cid, "score_0_1": score, "score_0_3": score * 3 if score is not None else None, "status": mdms.get("status"), "coverage": mdms.get("coverage"), "components": components, "mdms": {**mdms, "runtime_weights": config["weights"], "weights_metadata": {"version": config["version"], "selected_on": config["selection_scope"], "blind_evaluated": config["blind_evaluated"]}}}
            result["_preparation"] = {"cv": cv["manifest"], "jd": jd["manifest"]}
            results.append(result)
    except RuntimePreparationError as exc:
        raise RuntimeMatchingError(str(exc)) from exc
    match_run_id = uuid.uuid4().hex
    root = Path(settings.RUNTIME_DATA_DIR).parent / "matching" / match_run_id; (root / "candidates").mkdir(parents=True, exist_ok=True)
    clean_results = []
    for result in results:
        persisted = {k: v for k, v in result.items() if k != "_preparation"}
        (root / "candidates" / f"{result['cv_id']}.json").write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
        clean_results.append(persisted)
    manifest = {"schema_version":"runtime_matching_v1", "match_run_id":match_run_id, "domain":domain, "job":{"run_id":jrun,"document_id":jid,"runtime_input_sha256":jd["manifest"]["runtime_input_sha256"]}, "candidates":[{"run_id":r,"document_id":d,"runtime_input_sha256":next(x["_preparation"]["cv"]["runtime_input_sha256"] for x in results if x["cv_id"]==d)} for r,d in refs], "mdms_config":{"version":config["version"],"weights":config["weights"],"selected_on":config["selection_scope"],"blind_evaluated":config["blind_evaluated"]}, "embedding":{"model_name":jd["manifest"]["embedding"]["model_name"],"dimension":jd["manifest"]["embedding"]["dimension"]}}
    (root / "match_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"match_run_id":match_run_id, "domain":domain, "job":job_ref, "results":clean_results, "manifest":manifest}

def _config_for_matchers(config: dict[str, Any]) -> dict[str, Any]:
    import yaml
    path = Path(__file__).resolve().parents[3] / "configs" / "mdms.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
