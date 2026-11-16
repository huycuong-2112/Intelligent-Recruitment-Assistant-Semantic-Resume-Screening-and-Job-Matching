from app.frontend.utils.runtime_ranking import build_ranking_rows
def test_ranking_sort_zero_none_and_identity_without_mutating():
    response={'match_run_id':'m','results':[{'cv_id':'cv_b','score_0_1':None,'score_0_3':None,'status':'insufficient_data','components':{}},{'cv_id':'cv_a','score_0_1':0.0,'score_0_3':0.0,'status':'evaluated','components':{'skill':{'score':0.0}}},{'cv_id':'cv_c','score_0_1':.7,'score_0_3':2.1,'status':'evaluated','components':{}}]}
    rows=build_ranking_rows(response,{'cv_a':'A.pdf','cv_b':'B.pdf','cv_c':'C.pdf'})
    assert [r['document_id'] for r in rows]==['cv_c','cv_a','cv_b']
    assert rows[1]['score_0_1']==0.0 and rows[2]['rank'] is None and response['results'][0]['cv_id']=='cv_b'
    assert set(rows[0]) >= {'skill_score','experience_score','education_score','semantic_score'} and 'domain_score' not in rows[0]
