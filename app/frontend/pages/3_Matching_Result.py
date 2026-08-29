import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import streamlit as st
from app.frontend.components.navbar import render_navbar
from app.frontend.utils.api_client import run_runtime_matching, ResumeParseAPIError
from app.frontend.utils.runtime_matching import validate_runtime_matching_state, map_runtime_results
from app.frontend.utils.presentation import group_by_category

ACTIVE_DOMAIN = "IT"
render_navbar("Rating")
st.title("🔍 Kết quả Matching")
cv_states = st.session_state.get("cv_confirmed_candidates") or []
jd_state = st.session_state.get("jd_confirmed_document")
raw_jd = st.session_state.get("jd_features") or []
raw_cvs = st.session_state.get("cv_candidates") or []

def current_jd_features():
    return [f for i, f in enumerate(raw_jd) if st.session_state.get(f"jd_feat_{i}", True)]
def current_cv_features(candidate):
    raw = candidate.get("raw_features") or []; idx = raw_cvs.index(candidate) if candidate in raw_cvs else 0
    return [f for j, f in enumerate(raw) if st.session_state.get(f"cv_{idx}_feat_{j}", True)]

preflight = validate_runtime_matching_state(jd_state, current_jd_features(), cv_states, {c.get("document_id"): current_cv_features(c) for c in raw_cvs})
if st.session_state.get("runtime_match_response") and not preflight["ready"]:
    st.session_state["runtime_match_response"] = None; st.session_state["rating_results"] = None; st.session_state["match_run_id"] = None
if not jd_state or not cv_states:
    st.warning("Vui lòng hoàn tất Upload → Extract → Refine → Confirm ở cả CV và Job Description trước."); st.stop()
with st.expander("📋 Feature JD đã xác nhận"): st.write(group_by_category(jd_state.get("confirmed_features") or []))
st.subheader(f"👥 {len(cv_states)} CV sẽ được đánh giá")
for candidate in cv_states:
    with st.expander(f"📄 {candidate.get('filename', candidate.get('document_id'))}", expanded=False): st.write(group_by_category(candidate.get("confirmed_features") or []))
if not preflight["ready"]:
    for issue in preflight["issues"]: st.warning(issue)
if st.button("🚀 Chạy Matching cho tất cả CV", type="primary", use_container_width=True, disabled=not preflight["ready"]):
    with st.spinner("Đang chạy Matching MDMS cho tất cả CV..."):
        try:
            response = run_runtime_matching(ACTIVE_DOMAIN, preflight["job_ref"], preflight["candidate_refs"])
            by_id = {c.get("document_id"): c.get("filename", c.get("document_id")) for c in cv_states}
            run_ids = {c.get("document_id"): c.get("run_id") for c in cv_states}
            st.session_state["runtime_match_response"] = response; st.session_state["match_run_id"] = response.get("match_run_id")
            st.session_state["rating_results"] = map_runtime_results(response, by_id, run_ids)
        except ResumeParseAPIError as exc: st.error(str(exc))
results = st.session_state.get("rating_results") or []
if results:
    ordered = sorted(enumerate(results), key=lambda p: (p[1].get("score_0_1") is not None, p[1].get("score_0_1") if p[1].get("score_0_1") is not None else -1, -p[0]), reverse=True)
    st.divider(); st.subheader("🏆 Kết quả MDMS")
    for rank, (_, result) in enumerate(ordered, 1):
        score = result.get("score_0_3"); text = f"{score:.2f} / 3.00" if score is not None else "Insufficient data"
        cols = st.columns([1, 5, 2]); cols[0].markdown(f"### #{rank}"); cols[1].markdown(f"**📄 {result.get('filename', result.get('cv_id'))}**"); cols[2].metric("MDMS", text)
        if result.get("score_0_1") is not None: cols[2].caption(f"0–1: {result['score_0_1']:.3f}")
        with st.expander("Chi tiết thành phần"):
            for label in ("skill", "experience", "education", "semantic"):
                component = result.get(label) or {}; value = component.get("score"); status = component.get("status") or component.get("availability") or "unknown"
                shown = "N/A" if value is None and status in {"not_required", "not_applicable"} else "Unknown" if value is None else f"{value:.3f}"
                st.write(f"{label.title()}: {shown} ({status})")
    evaluated = [r for _, r in ordered if r.get("score_0_1") is not None]
    if evaluated:
        st.divider(); st.metric("Điểm trung bình (0–3)", f"{sum(r['score_0_3'] for r in evaluated)/len(evaluated):.2f} / 3.00")
    st.caption(f"Match run: {st.session_state.get('match_run_id')}")
