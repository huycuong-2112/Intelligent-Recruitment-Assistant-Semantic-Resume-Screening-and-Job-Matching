from app.frontend.utils.runtime_explanation import format_report_score, report_dimensions, report_weights, resolve_report_evidence


def test_report_projection_is_null_safe_and_preserves_canonical_scores():
    report={"decision":{"final_score":0.4233333333,"dimensions":{"skill":0.216,"experience":None,"education":0.6,"semantic":0.579},"weights":{"skill":.4}}}
    assert format_report_score(report["decision"]["final_score"]) == "1.27 / 3.00"
    assert report_dimensions(report)["skill"] == .216
    assert report_dimensions(report)["experience"] is None
    assert report_weights(report)["skill"] == .4
    assert "st.json" not in open("app/frontend/pages/4_Candidate_Ranking.py", encoding="utf-8").read()


def test_evidence_resolution_hides_unresolved_ids_from_primary_projection():
    report={"selected_evidence":{"ev_1":{"source_type":"CV","source_text":"Built APIs"}}}
    assert resolve_report_evidence(report,["ev_1"])[0]["source_text"] == "Built APIs"
    assert resolve_report_evidence(report,["ev_missing"]) == []
