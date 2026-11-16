import pytest

from app.frontend.utils import api_client

def test_runtime_matching_identity_only(monkeypatch):
    seen={}
    class Response:
        status_code=200
        def json(self): return {"match_run_id":"m","results":[]}
    def post(url, **kwargs): seen.update(url=url, kwargs=kwargs); return Response()
    monkeypatch.setattr(api_client.requests, "post", post)
    api_client.run_runtime_matching("IT", {"run_id":"a","document_id":"jd_1","runtime_parsed":{}}, [{"run_id":"b","document_id":"cv_1","confirmed_features":[]}])
    assert seen["url"].endswith("/matching/run")
    assert seen["kwargs"]["json"] == {"domain":"IT","job":{"run_id":"a","document_id":"jd_1"},"candidates":[{"run_id":"b","document_id":"cv_1"}]}

def test_generate_candidate_explanation_identity_only(monkeypatch):
    seen={}
    class Response:
        status_code=200
        def json(self): return {"schema_version":"explanation_v1"}
    monkeypatch.setattr(api_client.requests,"post",lambda url,**kw:(seen.update(url=url,kw=kw) or Response()))
    api_client.generate_candidate_explanation("a","cv_1")
    assert seen["url"].endswith("/explanations/generate")
    assert seen["kw"]["json"]=={"match_run_id":"a","cv_id":"cv_1","mode":"auto"}


class Upload:
    name = "candidate.pdf"
    type = "application/pdf"

    def getvalue(self):
        return b"pdf-bytes"


class Response:
    def __init__(self, status, payload): self.status_code, self.payload = status, payload
    def json(self): return self.payload


def test_upload_resume_sends_real_multipart_and_offline(monkeypatch):
    seen = {}
    def post(url, **kwargs):
        seen.update(url=url, **kwargs); return Response(200, {"run_id": "r", "parsed": {}, "ui_features": []})
    monkeypatch.setattr(api_client.requests, "post", post)
    assert api_client.upload_resume(Upload(), offline=True)["run_id"] == "r"
    assert seen["url"].endswith("/resume/parse")
    assert seen["params"] == {"offline": "true"}
    assert seen["files"]["file"] == ("candidate.pdf", b"pdf-bytes", "application/pdf")


def test_upload_resume_surfaces_detail_and_connection_error(monkeypatch):
    monkeypatch.setattr(api_client.requests, "post", lambda *a, **k: Response(415, {"detail": "PDF only"}))
    with pytest.raises(api_client.ResumeParseAPIError, match="PDF only"): api_client.upload_resume(Upload())
    def fail(*a, **k): raise api_client.requests.RequestException("down")
    monkeypatch.setattr(api_client.requests, "post", fail)
    with pytest.raises(api_client.ResumeParseAPIError, match="Backend unavailable"): api_client.upload_resume(Upload())

def test_confirm_clients_send_only_identity_and_features(monkeypatch):
    seen = {}
    monkeypatch.setattr(api_client.requests, "post", lambda url, **kw: (seen.update(url=url, **kw) or Response(200, {"status":"APPLIED"})))
    api_client.confirm_resume("r", "cv_1", [{"id":"f"}]); assert seen["url"].endswith("/resume/confirm"); assert set(seen["json"]) == {"run_id","document_id","confirmed_features"}
    api_client.confirm_job("r", "jd_1", []); assert seen["url"].endswith("/job/confirm")
