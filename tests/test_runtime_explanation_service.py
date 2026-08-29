import pytest
from app.api.services.runtime_explanation_service import generate_runtime_explanation
from app.api.services.runtime_xai_service import RuntimeXAIError
from app.api.schemas.explanation_schema import ExplanationRequest

RUN='a'*32
def test_identity_schema_forbids_spoofed_scores():
    with pytest.raises(Exception): ExplanationRequest(match_run_id=RUN,cv_id='cv_1',mode='offline',final_score=1)

def test_modes_and_offline_none_safe(monkeypatch):
    # use a compact controlled xai and bypass filesystem bridge for mode guards
    xai={'schema_version':'xai_v1','jd_id':'jd_1','cv_id':'cv_1','job_title':None,'decision':{'final_score':None,'status':'insufficient_data','coverage':0.0,'weights':{'skill':.4,'experience':.2,'education':.1,'semantic':.3},'effective_weights':{},'model_version':'mdms_runtime_v1'},'dimensions':{k:{'score':None,'status':'UNKNOWN','weight':.25,'weighted_contribution':None,'coverage':0.0} for k in ('skill','experience','education','semantic')},'strength_candidates':[],'gap_candidates':[],'interview_focus':[],'evidence_registry':{}}
    monkeypatch.setattr('app.api.services.runtime_explanation_service.build_runtime_xai',lambda *a:xai)
    out=generate_runtime_explanation(RUN,'cv_1','offline')
    assert out['decision']['final_score'] is None and out['generation']['method']=='offline_deterministic'
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    assert generate_runtime_explanation(RUN,'cv_1','auto')['generation']['fallback_used'] is False
    with pytest.raises(RuntimeXAIError): generate_runtime_explanation(RUN,'cv_1','groq')
