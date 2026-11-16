"""Pure frontend preflight and presentation helpers for runtime matching."""
from __future__ import annotations
from typing import Any
from .state_utils import is_runtime_ready

def validate_runtime_matching_state(jd_state: dict[str, Any] | None, jd_current_features: list[dict] | None,
                                    cv_states: list[dict[str, Any]], cv_current_features: dict[str, list[dict]]) -> dict[str, Any]:
    issues=[]; job_ref=None; candidate_refs=[]
    if not jd_state or not is_runtime_ready(jd_state, jd_current_features):
        issues.append("Job Description has unconfirmed changes. Confirm it before matching.")
    else:
        job_ref={"run_id":jd_state.get("run_id"),"document_id":jd_state.get("document_id")}
    for state in cv_states:
        did=state.get("document_id"); current=cv_current_features.get(did, [])
        if not is_runtime_ready(state, current):
            issues.append(f"Confirm CV {state.get('filename') or did} before matching.")
        else: candidate_refs.append({"run_id":state.get("run_id"),"document_id":did})
    if issues: return {"ready":False,"job_ref":None,"candidate_refs":[],"issues":issues}
    return {"ready":bool(job_ref and candidate_refs),"job_ref":job_ref,"candidate_refs":candidate_refs,"issues":[] if candidate_refs else ["Confirm at least one CV before matching."]}

def map_runtime_results(response: dict[str, Any], filename_by_id: dict[str, str], run_id_by_id: dict[str, str] | None = None) -> list[dict[str, Any]]:
    run_id=response.get("match_run_id"); mapped=[]
    for item in response.get("results", []):
        did=item.get("cv_id") or item.get("document_id")
        mapped.append({"filename":filename_by_id.get(did, did), "run_id":(run_id_by_id or {}).get(did), "document_id":did, "cv_id":did,
                       "score_0_1":item.get("score_0_1"), "score_0_3":item.get("score_0_3"), "status":item.get("status"), "coverage":item.get("coverage"),
                       **{k:item.get("components",{}).get(k) for k in ("skill","experience","education","semantic")}, "mdms":item.get("mdms"), "match_run_id":run_id})
    return mapped
