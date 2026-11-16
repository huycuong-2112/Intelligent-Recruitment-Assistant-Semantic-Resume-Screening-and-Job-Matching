from app.frontend.utils.runtime_matching import validate_runtime_matching_state, map_runtime_results
from app.frontend.utils.state_utils import feature_selection_fingerprint

def state(run, doc, fp='x', status='APPLIED'):
    return {'run_id':run,'document_id':doc,'runtime_parsed':{'id':doc},'confirm_status':status,'confirmed_feature_fingerprint':fp}
def test_preflight_blocks_dirty_and_allows_partial():
    fp=feature_selection_fingerprint([]); jd=state('a'*32,'jd_1',fp); cv=state('b'*32,'cv_1',fp,'PARTIAL')
    assert validate_runtime_matching_state(jd,[],[cv],{'cv_1':[]})['ready']
    assert not validate_runtime_matching_state(jd,[{'id':'changed'}],[cv],{'cv_1':[]})['ready']
def test_result_mapping_uses_document_id_and_zero_none():
    resp={'match_run_id':'m','results':[{'cv_id':'cv_b','score_0_1':0.0,'score_0_3':0.0,'status':'evaluated','components':{}},{'cv_id':'cv_a','score_0_1':None,'score_0_3':None,'status':'insufficient_data','components':{}}]}
    out=map_runtime_results(resp,{'cv_a':'A.pdf','cv_b':'B.pdf'})
    assert out[0]['filename']=='B.pdf' and out[0]['score_0_1']==0.0 and out[1]['filename']=='A.pdf'
