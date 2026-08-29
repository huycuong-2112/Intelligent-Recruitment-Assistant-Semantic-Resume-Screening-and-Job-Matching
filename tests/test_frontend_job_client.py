import pytest
from app.frontend.utils import api_client

class Upload:
    name = "job.pdf"; type = "application/pdf"
    def getvalue(self): return b"pdf"
class Response:
    def __init__(self, status, data): self.status_code, self.data = status, data
    def json(self): return self.data

def test_upload_job_uses_job_multipart_endpoint(monkeypatch):
    seen = {}
    def post(url, **kwargs): seen.update(url=url, **kwargs); return Response(200, {"run_id": "r"})
    monkeypatch.setattr(api_client.requests, "post", post)
    assert api_client.upload_job(Upload(), offline=True)["run_id"] == "r"
    assert seen["url"].endswith("/job/parse") and seen["params"] == {"offline": "true"}
    assert seen["files"]["file"][0] == "job.pdf"

def test_upload_job_surfaces_detail(monkeypatch):
    monkeypatch.setattr(api_client.requests, "post", lambda *a, **k: Response(415, {"detail": "PDF only"}))
    with pytest.raises(api_client.ResumeParseAPIError, match="PDF only"): api_client.upload_job(Upload())
