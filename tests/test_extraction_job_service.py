import time
from app.api.services import extraction_job_service as manager

def wait(job_id, timeout=2):
    end=time.time()+timeout
    while time.time()<end:
        meta=manager.get(job_id)
        if meta and meta.get("status") in {"completed","failed"}: return meta
        time.sleep(.01)
    return manager.get(job_id)

def test_submit_is_immediate_and_persists_completed_result():
    started=time.time(); ref=manager.submit("resume","a.pdf",lambda:{"run_id":"r","document_id":"cv_1","extraction":{"status":"ACCEPTED_BY_DOCLING"},"parsed":{"extraction_method":"groq_llm"}})
    assert time.time()-started < .2 and ref["status"] == "queued"
    meta=wait(ref["job_id"]); assert meta["status"] == "completed" and meta["source_status"] == "ACCEPTED_BY_DOCLING"

def test_two_workers_overlap_and_third_is_queued_or_running():
    active=[]; peak=[0]
    def work():
        active.append(1); peak[0]=max(peak[0],len(active)); time.sleep(.08); active.pop(); return {}
    refs=[manager.submit("resume",f"{i}.pdf",work) for i in range(3)]
    metas=[wait(r["job_id"]) for r in refs]
    assert all(m["status"] == "completed" for m in metas) and peak[0] <= 2

def test_failed_job_isolated_with_safe_category():
    ref=manager.submit("job_description","bad.pdf",lambda: (_ for _ in ()).throw(RuntimeError("provider detail")))
    meta=wait(ref["job_id"]); assert meta["status"] == "failed" and meta["failure_category"] == "provider_error"
