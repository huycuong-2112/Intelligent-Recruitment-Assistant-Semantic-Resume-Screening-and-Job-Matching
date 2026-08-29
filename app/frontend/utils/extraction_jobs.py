"""Lightweight frontend references for backend-owned extraction jobs."""
from __future__ import annotations

def start_jobs(files, submit_fn):
    jobs = {}
    for order, file in enumerate(files):
        key = f"{file.name}:{getattr(file, 'size', 0)}:{order}"
        jobs[key] = {"key": key, "order": order, "filename": file.name, "status": "pending", "job_id": None, "file": file}
    return jobs, submit_fn

def submit_next(jobs, submit_fn):
    if not isinstance(jobs, dict) or not jobs: return False
    active = [j for j in jobs.values() if isinstance(j, dict)]
    if any(j.get("status") in {"queued", "running"} for j in active): return False
    pending = sorted((j for j in active if j.get("status") == "pending"), key=lambda x: x.get("order", 0))
    if not pending: return False
    job = pending[0]
    try:
        response = submit_fn(job["file"]); job["job_id"] = response["job_id"]; job["status"] = response.get("status", "queued")
    except Exception as exc: job["status"] = "failed"; job["error"] = str(exc)[:300]
    return True

def collect(jobs, callbacks):
    if not isinstance(jobs, dict) or not jobs: return False
    changed = submit_next(jobs, callbacks[0])
    for job in list(jobs.values()):
        if not isinstance(job, dict) or job.get("status") not in {"queued", "running"} or not job.get("job_id"): continue
        try: remote = callbacks[1](job["job_id"])
        except Exception as exc: job["status"] = "failed"; job["error"] = str(exc)[:300]; changed = True; continue
        state = remote.get("status")
        if state and state != job.get("status"): job["status"] = state; changed = True
        if state == "completed": job["result"] = remote.get("result"); changed = True
        if state == "failed": job["error"] = remote.get("error")
    if not any(isinstance(j, dict) and j.get("status") in {"queued", "running"} for j in jobs.values()):
        changed = submit_next(jobs, callbacks[0]) or changed
    return changed

def public_job(job): return {k:v for k,v in job.items() if k != "file"}
