import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")


class ResumeParseAPIError(RuntimeError):
    """User-safe error raised when the resume parsing API cannot be used."""

    def __init__(self, message: str, *, fatal: bool = True):
        super().__init__(message)
        self.fatal = fatal


def upload_resume(file, offline: bool | None = None) -> dict:
    filename = getattr(file, "name", None) or ""
    content = file.getvalue() if hasattr(file, "getvalue") else file.read()
    content_type = getattr(file, "type", None) or "application/octet-stream"
    try:
        response = requests.post(
            f"{API_BASE_URL}/resume/parse",
            params={} if offline is None else {"offline": str(bool(offline)).lower()},
            files={"file": (filename, content, content_type)},
            timeout=180,
        )
    except requests.RequestException as exc:
        raise ResumeParseAPIError("Backend unavailable. Start the FastAPI server and try again.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Resume parsing failed")
        except (ValueError, AttributeError):
            detail = "Resume parsing failed"
        if response.status_code >= 500:
            raise ResumeParseAPIError(f"CV extraction failed on the server: {detail}")
        raise ResumeParseAPIError(f"CV extraction could not process this document: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise ResumeParseAPIError("Backend returned an invalid parsing response") from exc

def upload_job(file, offline: bool | None = None) -> dict:
    filename = getattr(file, "name", "") or ""
    content = file.getvalue() if hasattr(file, "getvalue") else file.read()
    content_type = getattr(file, "type", None) or "application/octet-stream"
    try:
        params = {} if offline is None else {"offline": str(bool(offline)).lower()}
        response = requests.post(f"{API_BASE_URL}/job/parse", params=params, files={"file": (filename, content, content_type)}, timeout=180)
    except requests.RequestException as exc:
        raise ResumeParseAPIError("Backend unavailable. Start the FastAPI server and try again.") from exc
    if response.status_code >= 400:
        try: detail = response.json().get("detail", "Job parsing failed")
        except (ValueError, AttributeError): detail = "Job parsing failed"
        raise ResumeParseAPIError(str(detail))
    try: return response.json()
    except ValueError as exc: raise ResumeParseAPIError("Backend returned an invalid parsing response") from exc

def submit_resume_extraction(file, offline: bool | None = None) -> dict:
    return _submit_extraction("resume/parse/async", file, offline)

def submit_job_extraction(file, offline: bool | None = None) -> dict:
    return _submit_extraction("job/parse/async", file, offline)

def _submit_extraction(path, file, offline):
    filename=getattr(file,"name","") or ""; content=file.getvalue() if hasattr(file,"getvalue") else file.read(); content_type=getattr(file,"type",None) or "application/octet-stream"
    try: response=requests.post(f"{API_BASE_URL}/{path}",params={} if offline is None else {"offline":str(bool(offline)).lower()},files={"file":(filename,content,content_type)},timeout=30)
    except requests.RequestException as exc: raise ResumeParseAPIError("Backend unavailable. Start the FastAPI server and try again.") from exc
    if response.status_code >= 400: raise ResumeParseAPIError(str(response.json().get("detail","Extraction submission failed")))
    return response.json()

def get_extraction_status(kind: str, job_id: str) -> dict:
    try: response=requests.get(f"{API_BASE_URL}/{kind}/jobs/{job_id}",timeout=10)
    except requests.RequestException as exc: raise ResumeParseAPIError("Backend unavailable. Start the FastAPI server and try again.", fatal=False) from exc
    if response.status_code == 404: raise ResumeParseAPIError("Extraction job not found", fatal=True)
    if response.status_code >= 400: raise ResumeParseAPIError("Extraction status temporarily unavailable", fatal=False)
    return response.json()

def _confirm(path: str, run_id: str, document_id: str, confirmed_features: list[dict]) -> dict:
    try:
        response = requests.post(f"{API_BASE_URL}/{path}", json={"run_id": run_id, "document_id": document_id, "confirmed_features": confirmed_features}, timeout=180)
    except requests.RequestException as exc:
        raise ResumeParseAPIError("Backend unavailable. Start the FastAPI server and try again.") from exc
    if response.status_code >= 400:
        try: detail = response.json().get("detail", "Confirmation failed")
        except (ValueError, AttributeError): detail = "Confirmation failed"
        raise ResumeParseAPIError(str(detail))
    try: return response.json()
    except ValueError as exc: raise ResumeParseAPIError("Backend returned an invalid confirmation response") from exc

def confirm_resume(run_id: str, document_id: str, confirmed_features: list[dict]) -> dict:
    return _confirm("resume/confirm", run_id, document_id, confirmed_features)

def confirm_job(run_id: str, document_id: str, confirmed_features: list[dict]) -> dict:
    return _confirm("job/confirm", run_id, document_id, confirmed_features)

def run_runtime_matching(domain: str, job: dict, candidates: list[dict]) -> dict:
    """Run structured server-authoritative matching with identity references only."""
    payload = {"domain": domain, "job": {"run_id": job["run_id"], "document_id": job["document_id"]},
               "candidates": [{"run_id": c["run_id"], "document_id": c["document_id"]} for c in candidates]}
    try:
        response = requests.post(f"{API_BASE_URL}/matching/run", json=payload, timeout=600)
    except requests.RequestException as exc:
        raise ResumeParseAPIError("Backend unavailable. Start the FastAPI server and try again.") from exc
    if response.status_code >= 400:
        try: detail = response.json().get("detail", "Runtime matching failed")
        except (ValueError, AttributeError): detail = "Runtime matching failed"
        raise ResumeParseAPIError(str(detail))
    try: return response.json()
    except ValueError as exc: raise ResumeParseAPIError("Backend returned an invalid matching response") from exc

def generate_candidate_explanation(match_run_id: str, cv_id: str, mode: str = "auto") -> dict:
    payload = {"match_run_id": match_run_id, "cv_id": cv_id, "mode": mode}
    try:
        response = requests.post(f"{API_BASE_URL}/explanations/generate", json=payload, timeout=600)
    except requests.RequestException as exc:
        raise ResumeParseAPIError("Explanation service is unavailable.") from exc
    if response.status_code >= 400:
        try: detail = response.json().get("detail", "Explanation generation failed")
        except (ValueError, AttributeError): detail = "Explanation generation failed"
        raise ResumeParseAPIError(str(detail))
    try: return response.json()
    except ValueError as exc: raise ResumeParseAPIError("Backend returned an invalid explanation response") from exc
