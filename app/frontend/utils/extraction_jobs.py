"""Lightweight frontend references for backend-owned extraction jobs."""
from __future__ import annotations
import time

ACTIVE_STATES = {"queued", "running", "processing", "long_running"}
TERMINAL_STATES = {"completed", "failed", "removed"}
MAX_WAIT_SECONDS = 300

def start_jobs(files, submit_fn):
    jobs = {}
    for order, file in enumerate(files):
        key = f"{file.name}:{getattr(file, 'size', 0)}:{order}"
        jobs[key] = {"key": key, "order": order, "filename": file.name, "status": "pending", "job_id": None, "file": file, "submitted_at": None, "poll_errors": 0}
    return jobs, submit_fn

def submit_next(jobs, submit_fn):
    if not isinstance(jobs, dict) or not jobs: return False
    active = [j for j in jobs.values() if isinstance(j, dict)]
    if any(j.get("status") in ACTIVE_STATES for j in active): return False
    pending = sorted((j for j in active if j.get("status") == "pending"), key=lambda x: x.get("order", 0))
    if not pending: return False
    job = pending[0]
    try:
        response = submit_fn(job["file"]); job["job_id"] = response["job_id"]; job["status"] = response.get("status", "queued"); job["submitted_at"] = time.monotonic(); job["poll_errors"] = 0
    except Exception as exc: job["status"] = "failed"; job["error"] = str(exc)[:300]
    return True

def collect(jobs, callbacks):
    if not isinstance(jobs, dict) or not jobs: return False
    changed = submit_next(jobs, callbacks[0])
    for job in list(jobs.values()):
        if not isinstance(job, dict) or job.get("status") not in ACTIVE_STATES or not job.get("job_id"): continue
        try: remote = callbacks[1](job["job_id"])
        except Exception as exc:
            if getattr(exc, "fatal", False):
                job["status"] = "failed"; job["error"] = str(exc)[:300]; changed = True
            else:
                job["poll_errors"] = int(job.get("poll_errors") or 0) + 1
                job["last_error"] = str(exc)[:300]
            continue
        state = remote.get("status")
        if state and not (job.get("status") == "long_running" and state in ACTIVE_STATES) and state != job.get("status"):
            job["status"] = state; changed = True
        if state == "completed": job["result"] = remote.get("result"); changed = True
        if state == "failed": job["error"] = remote.get("error")
        if state in ACTIVE_STATES and job.get("submitted_at") is not None and time.monotonic() - job["submitted_at"] >= MAX_WAIT_SECONDS:
            if job.get("status") != "long_running": job["status"] = "long_running"; changed = True
    if not any(isinstance(j, dict) and j.get("status") in ACTIVE_STATES for j in jobs.values()):
        changed = submit_next(jobs, callbacks[0]) or changed
    return changed

def consume_completed_jobs(jobs, candidates=None):
    """Persist completed backend results before deactivating polling state."""
    by_filename = {c.get("filename"): c for c in (candidates or []) if isinstance(c, dict)}
    changed = False
    for job in (jobs or {}).values():
        if not isinstance(job, dict) or job.get("status") != "completed" or not isinstance(job.get("result"), dict):
            continue
        result = job["result"]
        filename = result.get("filename", job.get("filename"))
        if filename in by_filename:
            by_filename[filename].update({"refinable": True, "result": result})
            continue
        by_filename[filename] = {
            "filename": filename,
            "run_id": result.get("run_id"),
            "document_id": result.get("document_id"),
            "extraction": result.get("extraction", {}),
            "parsed": result.get("parsed"),
            "raw_features": result.get("ui_features", []),
            "refinable": True,
            "result": result,
        }
        changed = True
    ordered = sorted(by_filename.values(), key=lambda c: next((j.get("order", 0) for j in (jobs or {}).values() if j.get("filename") == c.get("filename")), 0))
    return ordered, changed

def public_job(job): return {k:v for k,v in job.items() if k != "file"}
