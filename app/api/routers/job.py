from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from app.api.schemas.job_schema import JobParseResponse, ConfirmRequest, ConfirmResponse
from app.api.services.confirm_service import ConfirmServiceError, confirm_document
from app.api.services.job_service import JobParseError, parse_uploaded_job
from app.api.services.extraction_job_service import submit as submit_extraction, get as get_extraction, public as public_extraction, result as extraction_result

router = APIRouter(prefix="/job", tags=["Job"])

@router.post("/parse/async", status_code=202)
async def submit_job_extraction(file: UploadFile = File(...), offline: bool | None = Query(None)):
    content = await file.read(); filename = file.filename or ""
    return submit_extraction("job_description", filename, lambda: parse_uploaded_job(filename, content, offline=offline))

@router.get("/jobs/{job_id}")
def job_extraction_status(job_id: str):
    meta = get_extraction(job_id)
    if not meta: raise HTTPException(status_code=404, detail="Extraction job not found")
    payload = public_extraction(meta)
    if meta.get("status") == "completed": payload["result"] = extraction_result(job_id)
    return payload


@router.get("/ping")
def ping():
    return {"message": "Job router is working"}


@router.post("/parse", response_model=JobParseResponse)
async def parse_job(file: UploadFile = File(...), offline: bool | None = Query(None)):
    try:
        return parse_uploaded_job(file.filename or "", await file.read(), offline=offline)
    except JobParseError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Job parsing failed")

@router.post("/confirm", response_model=ConfirmResponse)
def confirm_job(request: ConfirmRequest):
    try: return confirm_document(request.run_id, request.document_id, request.confirmed_features, "jd")
    except ConfirmServiceError as exc: raise HTTPException(status_code=exc.status_code, detail=str(exc))
