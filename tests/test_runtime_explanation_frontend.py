from app.frontend.utils.runtime_explanation import validate_report_matches_candidate, format_report_score, report_cache_key
def test_score_guard_and_display():
    e={'match_run_id':'m','cv_id':'cv','decision':{'final_score':.5}}
    assert validate_report_matches_candidate(e,'m',{'cv_id':'cv','score_0_1':.5})[0]
    assert format_report_score(None)=='Insufficient data' and format_report_score(0)=='0.00 / 3.00'
def test_identity_and_cache_guard():
    assert not validate_report_matches_candidate({'match_run_id':'other','cv_id':'cv','decision':{'final_score':.5}},'m',{'cv_id':'cv','score_0_1':.5})[0]
    assert report_cache_key('m','a') != report_cache_key('m','b')
