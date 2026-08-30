from pathlib import Path


PAGE = Path("app/frontend/pages/1_Upload_CV.py").read_text(encoding="utf-8")
JD_PAGE = Path("app/frontend/pages/2_Upload_JobDescription.py").read_text(encoding="utf-8")


def test_cv_refine_is_cv_owned_and_not_gated_by_jd_state():
    refine = PAGE.split("# Refine", 1)[1].split("if st.session_state.get(\"cv_extraction_jobs\")", 1)[0]
    assert 'if st.session_state.get("cv_candidates")' in refine
    assert "jd_document" not in refine
    assert "jd_confirmed" not in refine


def test_progressive_poller_reruns_full_app_after_ready_candidate():
    assert 'st.rerun(scope="app")' in PAGE
    assert "ThreadPoolExecutor" not in PAGE  # queue ownership remains in extraction_jobs


def test_long_running_poll_is_neutral_and_failed_is_terminal_error():
    assert 'job["status"] == "long_running"' in PAGE
    assert "Đang xử lý lâu hơn dự kiến" in PAGE
    assert 'job["status"] == "failed"' in PAGE
    assert "job.get('error'," in PAGE


def test_each_cv_card_renders_backend_provenance_independently():
    assert 'source_status = extraction.get("status")' in PAGE
    assert 'extraction_method = (cand.get("parsed") or {}).get("extraction_method")' in PAGE
    for value in ("ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR", "LOW_QUALITY", "offline_hybrid"):
        assert value in PAGE


def test_jd_page_does_not_touch_cv_queue_or_cv_candidates():
    assert "cv_extraction_jobs" not in JD_PAGE
    assert "cv_candidates" not in JD_PAGE
