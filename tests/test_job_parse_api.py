import json, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.api.main import app
from app.api.services import job_service

@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(job_service.settings, "RUNTIME_DATA_DIR", str(tmp_path / "resumes"))
    class P:
        def parse(self, path): return "AI Engineer Intern\nRequirements:\n- Python\nResponsibilities:\n- Build models", {"final_status":"ACCEPTED_BY_DOCLING","docling_score":.9,"ocr_triggered":False,"ocr_score":None}
    monkeypatch.setitem(sys.modules, "document_parser", type("M", (), {"get_document_parser": staticmethod(lambda:P())})())
    def parse(inp, out, offline=False):
        d=json.loads(Path(inp).read_text())[0]; x={"id":d["id"],"extraction_method":"offline_hybrid","parsed_data":{"job_title":"AI Engineer Intern","company_name":None,"job_overview":"Build models","min_experience_years":0.0,"required_degree":"Any","preferred_fields":[],"required_skills":["Python"],"preferred_skills":[],"responsibilities":["Build models"],"key_deliverables":[],"required_certifications":[]}}; return [x]
    monkeypatch.setitem(sys.modules, "jd_parser", type("M", (), {"parse_cleaned_jds": staticmethod(parse)})())

def test_job_parse_pdf_returns_canonical_and_features(fake_pipeline):
    r=TestClient(app).post('/api/v1/job/parse',files={'file':('job.pdf',b'pdf','application/pdf')}); assert r.status_code==200
    b=r.json(); assert b['document_id'].startswith('jd_'); assert b['parsed']['parsed_data']['required_skills']==['Python']; assert b['ui_features']

@pytest.mark.parametrize('name', ['job.png', 'job.jpg', 'job.jpeg'])
def test_job_parse_accepts_image_extensions(name, fake_pipeline):
    assert TestClient(app).post('/api/v1/job/parse', files={'file': (name, b'image')}).status_code == 200

@pytest.mark.parametrize('name',["job.docx","job.txt"])
def test_job_parse_rejects_noncanonical_formats(name,fake_pipeline):
    assert TestClient(app).post('/api/v1/job/parse',files={'file':(name,b'x')}).status_code==415

def test_job_parse_rejects_empty(fake_pipeline):
    assert TestClient(app).post('/api/v1/job/parse',files={'file':('job.pdf',b'')}).status_code==400
