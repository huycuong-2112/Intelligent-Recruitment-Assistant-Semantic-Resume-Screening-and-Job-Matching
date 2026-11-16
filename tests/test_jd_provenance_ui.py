from pathlib import Path


PAGE = Path("app/frontend/pages/2_Upload_JobDescription.py").read_text(encoding="utf-8")


def test_jd_page_renders_backend_source_status_without_inference():
    assert 'source_status = extraction.get("status")' in PAGE
    for status in ("ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR", "LOW_QUALITY"):
        assert status in PAGE
    assert 'extraction_method = (st.session_state["jd_document"].get("parsed") or {}).get("extraction_method")' in PAGE
    assert 'extraction_method == "offline_hybrid"' in PAGE


def test_jd_page_keeps_low_quality_document_refinable_and_reextract_invalidates_state():
    assert 'if st.session_state.get("jd_document"):' in PAGE
    assert 'st.session_state["jd_confirmed_document"] = None' in PAGE
    assert 'st.session_state["jd_confirmed_features"] = None' in PAGE
    assert 'st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None' in PAGE
    assert 'raise' not in PAGE.split('if st.session_state.get("jd_document"):')[1].split('st.divider()', 1)[0]
