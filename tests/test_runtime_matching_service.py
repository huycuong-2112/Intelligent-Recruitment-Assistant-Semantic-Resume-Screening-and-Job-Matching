import json
from pathlib import Path
import pytest
from app.api.core.config import settings
from app.api.services.runtime_matching_service import run_runtime_matching, RuntimeMatchingError
from src.Matching.mdms import aggregate_mdms

RUNJ='b'*32; RUNC='c'*32
class FakeEmb:
    model_name='fake'; dimension=3; normalize_embeddings=True
    def embed_batch(self, texts): return [[1.,0.,0.] for _ in texts]
    def embed_text(self, text): return [1.,0.,0.]

def make(tmp_path):
    settings.RUNTIME_DATA_DIR=str(tmp_path/'resumes')
    jb=tmp_path/'jobs'/RUNJ/'jobs'; cb=tmp_path/'resumes'/RUNC/'resumes'; jb.mkdir(parents=True); cb.mkdir(parents=True)
    jd={'id':'jd_runtime','parsed_data':{'job_title':'AI','required_skills':['Python'],'responsibilities':['Build systems.'],'min_experience_years':1,'required_degree':'Bachelor','preferred_fields':[]}}
    cv={'id':'cv_runtime','parsed_data':{'skills':['Python'],'experience_years':2,'work_experience':[{'description':'Build systems.'}],'education_degree':'Bachelor','education_field':'CS','projects':[]}}
    (jb/'runtime_parsed_jd.json').write_text(json.dumps(jd)); (jb/'confirm_override.json').write_text(json.dumps({'document_id':'jd_runtime'}))
    (cb/'runtime_parsed_cv.json').write_text(json.dumps(cv)); (cb/'confirm_override.json').write_text(json.dumps({'document_id':'cv_runtime'}))

def test_runtime_matching_uses_four_matchers_and_frozen_weights(tmp_path):
    make(tmp_path)
    out=run_runtime_matching('IT',{'run_id':RUNJ,'document_id':'jd_runtime'},[{'run_id':RUNC,'document_id':'cv_runtime'}],FakeEmb())
    r=out['results'][0]
    assert set(r['components'])=={'skill','experience','education','semantic'}
    assert r['mdms']['runtime_weights']=={'skill':.4,'experience':.2,'education':.1,'semantic':.3}
    assert r['score_0_3']==pytest.approx(r['score_0_1']*3)
    assert (Path(settings.RUNTIME_DATA_DIR).parent/'matching'/out['match_run_id']/'match_manifest.json').is_file()

def test_duplicate_and_cross_type_rejected(tmp_path):
    make(tmp_path)
    ref={'run_id':RUNC,'document_id':'cv_runtime'}
    with pytest.raises(RuntimeMatchingError): run_runtime_matching('IT',{'run_id':RUNJ,'document_id':'jd_runtime'},[ref,ref],FakeEmb())
    with pytest.raises(RuntimeMatchingError): run_runtime_matching('IT',{'run_id':RUNC,'document_id':'jd_runtime'},[ref],FakeEmb())

def test_aggregate_zero_unknown_and_not_applicable_semantics():
    base={k:{'score':1.0,'availability':'available','status':'matched','coverage':1.0} for k in ('skill','experience','education','semantic')}
    base['skill']['score']=0.0
    assert aggregate_mdms(base,{'skill':.4,'experience':.2,'education':.1,'semantic':.3})['final_score']==pytest.approx(.6)
    base['experience']={'score':None,'availability':'unknown','status':'unknown'}
    assert aggregate_mdms(base,{'skill':.4,'experience':.2,'education':.1,'semantic':.3})['final_score'] is None
    base['experience']={'score':None,'availability':'not_applicable','status':'not_required'}
    e=aggregate_mdms(base,{'skill':.4,'experience':.2,'education':.1,'semantic':.3})['effective_weights']
    assert e['skill']==pytest.approx(.5)
