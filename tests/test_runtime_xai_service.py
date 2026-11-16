import json
from pathlib import Path
import pytest
from app.api.core.config import settings
from app.api.services.runtime_xai_service import build_runtime_xai, RuntimeXAIError

def make_tree(tmp_path):
    settings.RUNTIME_DATA_DIR=str(tmp_path/'resumes'); mr='a'*32; cr='b'*32; jr='c'*32
    mroot=tmp_path/'matching'/mr; (mroot/'candidates').mkdir(parents=True); (mroot/'xai').mkdir()
    cvbase=tmp_path/'resumes'/cr/'resumes'/'prepared'; jbase=tmp_path/'jobs'/jr/'jobs'/'prepared'; cvbase.mkdir(parents=True); jbase.mkdir(parents=True)
    cv={'id':'cv_1','domain':'IT','profile':{},'skills':{'all':['Python'],'explicit':['Python'],'project_derived':[]},'experience':{'professional_years':1,'work_evidence':[],'project_evidence':[]},'education':{'degree':'Bachelor','field':'CS'},'projects':[]}
    jd={'id':'jd_1','domain':'IT','role':{'job_title':'Engineer','overview':'Build systems.'},'skills':{'required':['Python'],'preferred':[]},'experience':{'minimum_years':1},'education':{'minimum_degree':'Bachelor','preferred_fields':[]},'responsibilities':[]}
    for base,obj,kind,run,doc in ((cvbase,cv,'cv',cr,'cv_1'),(jbase,jd,'jd',jr,'jd_1')):
        (base/('normalized_cv.json' if kind=='cv' else 'normalized_jd.json')).write_text(json.dumps(obj)); (base/'preparation_manifest.json').write_text(json.dumps({'runtime_input_sha256':run+'hash'}))
    manifest={'match_run_id':mr,'job':{'run_id':jr,'document_id':'jd_1','runtime_input_sha256':jr+'hash'},'candidates':[{'run_id':cr,'document_id':'cv_1','runtime_input_sha256':cr+'hash'}]}
    (mroot/'match_manifest.json').write_text(json.dumps(manifest))
    result={'jd_id':'jd_1','cv_id':'cv_1','components':{'skill':{'score':1,'requirements':[]},'experience':{'score':1,'evidence':{'score':None,'status':'not_required','details':{'responsibilities':[]}},'years':{'score':1,'status':'satisfied'}},'education':{'score':1,'degree':{'status':'satisfied','details':{'required':'Bachelor','candidate':'Bachelor'}},'field':{'status':'not_required','details':{}}},'semantic':{'score':.8,'status':'evaluated'}},'mdms':{'final_score':.94,'status':'evaluated','coverage':1,'runtime_weights':{'skill':.4,'experience':.2,'education':.1,'semantic':.3},'effective_weights':{'skill':.4,'experience':.2,'education':.1,'semantic':.3}}}
    (mroot/'candidates'/'cv_1.json').write_text(json.dumps(result)); return mr,cr,mroot

def test_runtime_xai_evaluated_and_idempotent(tmp_path):
    mr,cr,root=make_tree(tmp_path); a=build_runtime_xai(mr,'cv_1'); b=build_runtime_xai(mr,'cv_1')
    assert a['schema_version']=='xai_v1' and a['decision']['final_score']==.94 and a['decision']['weights']['skill']==.4 and a==b
    assert (root/'xai'/'cv_1.json').is_file()

def test_wrong_candidate_and_hash_rejected(tmp_path):
    mr,cr,root=make_tree(tmp_path)
    with pytest.raises(RuntimeXAIError): build_runtime_xai(mr,'cv_missing')
    p=tmp_path/'resumes'/cr/'resumes'/'prepared'/'preparation_manifest.json'; p.write_text(json.dumps({'runtime_input_sha256':'different'}))
    with pytest.raises(RuntimeXAIError): build_runtime_xai(mr,'cv_1')

def test_path_traversal_rejected(tmp_path):
    with pytest.raises(RuntimeXAIError): build_runtime_xai('../'+'a'*32,'cv_1')
