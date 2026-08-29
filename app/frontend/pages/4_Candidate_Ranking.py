import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import pandas as pd
import streamlit as st
from app.frontend.components.charts import render_radar_chart
from app.frontend.components.navbar import render_navbar
from app.frontend.utils.runtime_ranking import build_ranking_rows
from app.frontend.utils.api_client import generate_candidate_explanation, ResumeParseAPIError
from app.frontend.utils.runtime_explanation import report_cache_key, validate_report_matches_candidate, format_report_score, report_dimensions, report_weights, resolve_report_evidence

st.set_page_config(layout="wide", page_title="Ranking & Analytics", page_icon="🏆")
render_navbar("Ranking", hr_only=True)
st.title("🏆 Candidate Ranking & 📊 Analytics Dashboard")
st.markdown("Xếp hạng dựa trên kết quả MDMS thực tế từ Matching Result.")

response=st.session_state.get("runtime_match_response")
if not response or not response.get("match_run_id"):
    st.warning("Please run Matching Result first."); st.stop()
candidates=st.session_state.get("cv_confirmed_candidates") or []
by_id={c.get("document_id"):c.get("filename",c.get("document_id")) for c in candidates}
rows=build_ranking_rows(response,by_id)
if st.session_state.get("ranking_source_match_run_id") != response.get("match_run_id"):
    st.session_state["ranking_source_match_run_id"]=response.get("match_run_id")
    st.session_state["ranking_rows"]=rows
else: rows=st.session_state.get("ranking_rows",rows)

st.info(f"Match run: {response['match_run_id'][:12]}… | {len(rows)} ứng viên")
table=[]
for r in rows:
    table.append({"Chọn":False,"Hạng":r["rank"] or "N/A","Candidate":r["filename"],"Document ID":r["document_id"],"MDMS / 3":r["score_0_3"],"Skill":r["skill_score"],"Experience":r["experience_score"],"Education":r["education_score"],"Semantic":r["semantic_score"],"Coverage":r["coverage"],"Status":r["status"] or "unknown"})
df=pd.DataFrame(table)
edited=st.data_editor(df, key="runtime_ranking_editor", column_config={"Chọn":st.column_config.CheckboxColumn(required=True),"MDMS / 3":st.column_config.NumberColumn(format="%.2f")}, disabled=[c for c in df.columns if c!="Chọn"], use_container_width=True, hide_index=True)

selected=edited[edited["Chọn"]==True]
if not selected.empty:
    st.subheader(f"✅ Shortlist ({len(selected)})")
    st.dataframe(selected.drop(columns=["Chọn"]),use_container_width=True,hide_index=True)
    st.download_button("📥 Tải shortlist (CSV)",selected.drop(columns=["Chọn"]).to_csv(index=False).encode("utf-8-sig"),"shortlist_ung_vien.csv","text/csv")

st.divider(); st.subheader("🔍 Chi tiết ứng viên")
names=[r["filename"] for r in rows]
if names:
    chosen=st.selectbox("Chọn ứng viên",names)
    selected_row=next(r for r in rows if r["filename"]==chosen)
    st.metric("MDMS Score", "Insufficient data" if selected_row["score_0_3"] is None else f"{selected_row['score_0_3']:.2f} / 3.00")
    vals={"Skill":selected_row["skill_score"],"Experience":selected_row["experience_score"],"Education":selected_row["education_score"],"Semantic":selected_row["semantic_score"]}
    if all(v is not None for v in vals.values()): render_radar_chart(vals,title=f"Radar — {chosen}")
    else: st.info("Radar unavailable because one or more dimensions are unknown/not applicable.")
    @st.dialog("📄 Candidate Assessment Report", width="large")
    def show_report(report):
        st.markdown(f"**Candidate:** {selected_row['filename']} — **Target role:** {report.get('target_role') or 'Unavailable'}")
        st.metric("MDMS", format_report_score((report.get("decision") or {}).get("final_score")))
        st.caption("Dimension scores use the canonical 0–1 scale; percentages are display-only.")
        dimensions=report_dimensions(report); weights=report_weights(report)
        for name, value in dimensions.items():
            label=name.title(); weight=weights.get(name)
            if value is None: st.write(f"{label}: N/A" + (f" · weight {float(weight):.0%}" if weight is not None else ""))
            else:
                st.write(f"{label}: {float(value):.1%}" + (f" · weight {float(weight):.0%}" if weight is not None else ""))
                st.progress(max(0.0, min(1.0, float(value))))
        narrative=report.get("explanation") or {}; st.subheader("Summary"); st.write(narrative.get("summary") or "No summary available.")
        for title,key in (("Strengths","strengths"),("Gaps","gaps"),("Interview Focus","interview_focus")):
            st.subheader(title)
            for item in narrative.get(key,[]):
                text=item.get("text") or item.get("question") or item.get("topic") or ""
                st.markdown(f"- {text}")
                evidence=resolve_report_evidence(report, item.get("evidence_refs"))
                for ev in evidence:
                    snippet=ev.get("source_text") or ev.get("source_name") or ev.get("source_type")
                    if snippet: st.caption(f"Evidence: {snippet}")
                refs=item.get("evidence_refs") or []
                if refs:
                    with st.expander("Technical details", expanded=False): st.code(", ".join(str(ref) for ref in refs))
                if key == "interview_focus":
                    reason=item.get("reason")
                    if reason: st.caption(f"Reason: {reason}")
        generation=report.get("generation") or {}; method=generation.get("method")
        provenance="Groq" if method == "groq_llm" else "Offline fallback" if generation.get("fallback_used") or method == "offline_deterministic" else method or "Unavailable"
        st.caption(f"Explanation: {provenance} | Match run: {report.get('match_run_id')}")
        st.info(narrative.get("disclaimer", ""))
    cache=st.session_state.setdefault("candidate_reports", {})
    key=report_cache_key(response.get("match_run_id"), selected_row["document_id"])
    if st.button("📝 Generate Report", type="secondary"):
        if key not in cache:
            try:
                report=generate_candidate_explanation(response["match_run_id"],selected_row["document_id"],mode="auto")
                ok,error=validate_report_matches_candidate(report,response["match_run_id"],selected_row)
                if not ok: st.error(error or "Report does not match current result.")
                else: cache[key]=report; show_report(report)
            except ResumeParseAPIError as exc: st.error(str(exc))
        else: show_report(cache[key])

st.divider(); st.header("📊 Analytics Dashboard")
st.info("Ablation analysis is experiment-only and is not generated by this runtime match.")
weights=((response.get("results") or [{}])[0].get("mdms") or {}).get("runtime_weights")
if weights:
    st.subheader("Runtime MDMS weights (frozen v1)")
    st.dataframe(pd.DataFrame([{"Dimension":k.title(),"Weight":v} for k,v in weights.items()]),hide_index=True,use_container_width=True)
    meta=((response.get("results") or [{}])[0].get("mdms") or {}).get("weights_metadata") or {}
    st.caption(f"version: {meta.get('version')} | selected_on: {meta.get('selected_on')} | blind_evaluated: {meta.get('blind_evaluated')}")
evaluated=[r for r in rows if r["score_0_1"] is not None]
if evaluated:
    st.metric("Average MDMS (0–3)",f"{sum(r['score_0_3'] for r in evaluated)/len(evaluated):.2f} / 3.00")
st.session_state["ranking_df"]=edited
