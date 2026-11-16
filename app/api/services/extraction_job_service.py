"""In-process, persisted extraction jobs shared by CV and JD endpoints."""
from __future__ import annotations
import json, threading, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from app.api.core.config import settings

_ROOT = Path(settings.RUNTIME_DATA_DIR).parent / "extraction" / "jobs"; _ROOT.mkdir(parents=True, exist_ok=True)
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="extract")
_LOCK = threading.RLock(); _FUTURES: dict[str, Any] = {}
def _now(): return datetime.now(timezone.utc).isoformat()
def _path(job_id): return _ROOT / f"{job_id}.json"
def _save(meta):
    p=_path(meta["job_id"]); tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); tmp.replace(p)
for p in _ROOT.glob("ext_*.json"):
    try:
        m=json.loads(p.read_text(encoding="utf-8"))
        if m.get("status") in {"queued","running"}: m.update(status="failed",failure_category="interrupted",error="Job interrupted by backend restart",updated_at=_now()); _save(m)
    except Exception: pass
def _run(job_id, worker):
    with _LOCK: m=json.loads(_path(job_id).read_text()); m.update(status="running",started_at=_now(),updated_at=_now()); _save(m)
    try:
        out=worker()
        with _LOCK:
            m=json.loads(_path(job_id).read_text()); m.update(status="completed",result=out,run_id=out.get("run_id"),document_id=out.get("document_id"),source_status=(out.get("extraction") or {}).get("status"),extraction_method=(out.get("parsed") or {}).get("extraction_method"),completed_at=_now(),updated_at=_now()); _save(m)
    except Exception as exc:
        with _LOCK: m=json.loads(_path(job_id).read_text()); m.update(status="failed",failure_category="provider_error",error=str(exc)[:300],updated_at=_now()); _save(m)
def submit(kind: str, filename: str, worker: Callable[[], dict[str, Any]]):
    job_id="ext_"+uuid.uuid4().hex; meta={"job_id":job_id,"type":kind,"status":"queued","filename":filename,"created_at":_now()}
    with _LOCK: _save(meta); _FUTURES[job_id]=_EXECUTOR.submit(_run,job_id,worker)
    return {"job_id":job_id,"status":"queued"}
def get(job_id):
    try: return json.loads(_path(job_id).read_text(encoding="utf-8"))
    except (OSError,ValueError): return None
def public(meta): return {k:v for k,v in meta.items() if k!="result"}
def result(job_id):
    m=get(job_id); return (m or {}).get("result") if m and m.get("status")=="completed" else None
