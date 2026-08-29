import json, uuid
from pathlib import Path
from fastapi.testclient import TestClient
from app.api.main import app
from app.api.services import confirm_service
from app.api.adapters.presentation_adapter import parsed_cv_to_ui_features, parsed_jd_to_ui_features

def _artifact(tmp_path, kind, parsed):
    root=tmp_path/("resumes" if kind=='cv' else 'jobs')/('a'*32)/("resumes" if kind=='cv' else 'jobs'); root.mkdir(parents=True); (root/('parsed_resumes.json' if kind=='cv' else 'parsed_jds.json')).write_text(json.dumps([parsed])); return 'a'*32

def test_cv_confirm_is_server_authoritative_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm_service.settings,'RUNTIME_DATA_DIR',str(tmp_path/'resumes'))
    parsed={'id':'cv_1','parsed_data':{'skills':['Python'],'projects':[],'experience_years':2.0}}
    run=_artifact(tmp_path,'cv',parsed); feature=parsed_cv_to_ui_features(parsed)[0]; before=Path(tmp_path/'resumes'/run/'resumes'/'parsed_resumes.json').read_text()
    r=TestClient(app).post('/api/v1/resume/confirm',json={'run_id':run,'document_id':'cv_1','confirmed_features':[]}); assert r.status_code==200; assert r.json()['status']=='APPLIED'; assert r.json()['runtime_parsed']['parsed_data']['skills']==[]; assert Path(tmp_path/'resumes'/run/'resumes'/'parsed_resumes.json').read_text()==before

def test_jd_confirm_manual_and_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm_service.settings,'RUNTIME_DATA_DIR',str(tmp_path/'resumes'))
    parsed={'id':'jd_1','parsed_data':{'required_skills':['Python'],'preferred_skills':[],'responsibilities':[]}}
    run=_artifact(tmp_path,'jd',parsed); features=parsed_jd_to_ui_features(parsed); req={'run_id':run,'document_id':'jd_1','confirmed_features':features+[{'name':'PyTorch','category':'Preferred Skills','source_type':'manual_ui'}]}; r=TestClient(app).post('/api/v1/job/confirm',json=req); assert r.status_code==200; assert r.json()['runtime_parsed']['parsed_data']['preferred_skills']==['PyTorch']
    r=TestClient(app).post('/api/v1/job/confirm',json={'run_id':run,'document_id':'jd_1','confirmed_features':features+[{'name':'Master','category':'Education','source_type':'manual_ui'}]}); assert r.json()['status']=='PARTIAL' and r.json()['unsupported_actions']

def test_confirm_rejects_traversal_and_cross_type(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm_service.settings,'RUNTIME_DATA_DIR',str(tmp_path/'resumes'))
    r=TestClient(app).post('/api/v1/resume/confirm',json={'run_id':'../x','document_id':'cv_1','confirmed_features':[]}); assert r.status_code==400
    r=TestClient(app).post('/api/v1/job/confirm',json={'run_id':'a'*32,'document_id':'cv_1','confirmed_features':[]}); assert r.status_code==404
