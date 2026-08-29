import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.services import resume_service


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(resume_service.settings, "RUNTIME_DATA_DIR", str(tmp_path))
    class Parser:
        def parse(self, path):
            return "Skills\nPython\nProjects\nDemo", {"final_status": "ACCEPTED_BY_DOCLING", "docling_score": .9, "ocr_triggered": False, "ocr_score": None}
    monkeypatch.setitem(__import__('sys').modules, "document_parser", type("M", (), {"get_document_parser": staticmethod(lambda: Parser())})())
    def parse_cleaned(inp, out, offline=False):
        doc = json.loads(Path(inp).read_text())[0]
        result = {"id": doc["id"], "filename": doc["filename"], "extraction_method": "offline_hybrid", "parsed_data": {"skills": ["Python"], "projects": [], "work_experience": [], "experience_years": 0, "education_degree": None, "education_field": None}}
        Path(out).write_text(json.dumps([result]))
        return [result]
    monkeypatch.setitem(__import__('sys').modules, "LLM_parser", type("M", (), {"parse_cleaned_resumes": staticmethod(parse_cleaned)})())


def test_parse_endpoint_accepts_pdf_and_returns_canonical_and_features(fake_pipeline):
    client = TestClient(app)
    response = client.post("/api/v1/resume/parse", files={"file": ("candidate.pdf", b"pdf-bytes", "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "candidate.pdf"
    assert body["extraction"]["status"] == "ACCEPTED_BY_DOCLING"
    assert body["parsed"]["parsed_data"]["skills"] == ["Python"]
    assert body["ui_features"][0]["category"] == "Skills"


@pytest.mark.parametrize("filename", ["candidate.docx", "candidate.txt"])
def test_parse_endpoint_rejects_unsupported_formats(filename, fake_pipeline):
    response = TestClient(app).post("/api/v1/resume/parse", files={"file": (filename, b"data")})
    assert response.status_code == 415


def test_parse_endpoint_rejects_empty_upload(fake_pipeline):
    response = TestClient(app).post("/api/v1/resume/parse", files={"file": ("candidate.pdf", b"")})
    assert response.status_code == 400
