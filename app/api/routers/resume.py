from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from app.api.schemas.resume_schema import ResumeParseResponse, ConfirmRequest, ConfirmResponse
from app.api.services.confirm_service import ConfirmServiceError, confirm_document
from app.api.services.resume_service import ResumeParseError, parse_uploaded_cv
from app.api.services.extraction_job_service import submit as submit_extraction, get as get_extraction, public as public_extraction, result as extraction_result

router = APIRouter(prefix="/resume", tags=["Resume"])

@router.post("/parse/async", status_code=202)
async def submit_resume_extraction(file: UploadFile = File(...), offline: bool | None = Query(None)):
    content = await file.read(); filename = file.filename or ""
    return submit_extraction("resume", filename, lambda: parse_uploaded_cv(filename, content, offline=offline))

@router.get("/jobs/{job_id}")
def resume_extraction_status(job_id: str):
    meta = get_extraction(job_id)
    if not meta: raise HTTPException(status_code=404, detail="Extraction job not found")
    payload = public_extraction(meta)
    if meta.get("status") == "completed": payload["result"] = extraction_result(job_id)
    return payload


@router.get("/ping")
def ping():
    return {"message": "Resume router is working"}


@router.post("/parse", response_model=ResumeParseResponse)
async def parse_resume(file: UploadFile = File(...), offline: bool | None = Query(None)):
    try:
        content = await file.read()
        return parse_uploaded_cv(file.filename or "", content, offline=offline)
    except ResumeParseError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Resume parsing failed")

@router.post("/confirm", response_model=ConfirmResponse)
def confirm_resume(request: ConfirmRequest):
    try: return confirm_document(request.run_id, request.document_id, request.confirmed_features, "cv")
    except ConfirmServiceError as exc: raise HTTPException(status_code=exc.status_code, detail=str(exc))
