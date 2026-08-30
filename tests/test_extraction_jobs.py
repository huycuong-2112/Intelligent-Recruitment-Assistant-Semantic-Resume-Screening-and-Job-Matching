from app.frontend.utils import extraction_jobs
from app.frontend.utils.extraction_jobs import collect, start_jobs, submit_next

class File:
    def __init__(self, name): self.name, self.size = name, 1

def test_none_and_empty_queue_are_noops():
    assert submit_next(None, lambda f: {}) is False
    assert collect(None, (lambda f: {}, lambda j: {})) is False
    assert submit_next({}, lambda f: {}) is False

def test_three_job_backend_reference_sequence():
    submitted=[]; states={}
    def submit(f): submitted.append(f.name); jid=f"ext_{f.name}"; states[jid]="completed"; return {"job_id":jid,"status":"queued"}
    def status(jid): return {"status":states[jid],"result":{"filename":jid[4:]}}
    jobs,_=start_jobs([File("a"),File("b"),File("c")],submit)
    for _ in range(5): collect(jobs,(submit,status))
    assert submitted == ["a","b","c"]
    assert all(j["status"]=="completed" for j in jobs.values())
    assert collect(jobs,(submit,status)) is False

def test_removed_job_is_not_submitted():
    calls=[]; jobs,_=start_jobs([File("a"),File("b")],lambda f: calls.append(f.name) or {"job_id":"x","status":"queued"})
    jobs[next(k for k,v in jobs.items() if v["filename"]=="b")]["status"]="removed"
    def submit(f): calls.append(f.name); return {"job_id":"a","status":"completed"}
    collect(jobs,(submit,lambda j:{"status":"completed","result":{}}))
    assert calls == ["a"]

def test_long_running_job_can_complete_later():
    jobs, _ = start_jobs([File("slow.jpg")], lambda f: {"job_id": "ext-slow", "status": "running"})
    collect(jobs, (lambda f: {"job_id": "ext-slow", "status": "running"}, lambda _: {"status": "running"}))
    job = next(iter(jobs.values()))
    job["submitted_at"] -= extraction_jobs.MAX_WAIT_SECONDS + 1
    collect(jobs, (lambda f: {"job_id": "ext-slow", "status": "running"}, lambda _: {"status": "running"}))
    assert job["status"] == "long_running"
    collect(jobs, (lambda f: {"job_id": "ext-slow", "status": "running"}, lambda _: {"status": "completed", "result": {"ok": True}}))
    assert job["status"] == "completed" and job["result"] == {"ok": True}

def test_polling_timeout_keeps_job_id_and_is_neutral():
    jobs, _ = start_jobs([File("ocr.jpg")], lambda f: {"job_id": "ext-ocr", "status": "queued"})
    collect(jobs, (lambda f: {"job_id": "ext-ocr", "status": "queued"}, lambda _: {"status": "running"}))
    job = next(iter(jobs.values())); job["submitted_at"] -= extraction_jobs.MAX_WAIT_SECONDS + 1
    collect(jobs, (lambda f: {"job_id": "ext-ocr", "status": "queued"}, lambda _: {"status": "running"}))
    assert job["status"] == "long_running" and job["job_id"] == "ext-ocr" and "error" not in job

def test_transient_poll_error_does_not_fail_but_fatal_error_does():
    jobs, _ = start_jobs([File("a.jpg")], lambda f: {"job_id": "ext-a", "status": "queued"})
    def transient(_): raise RuntimeError("temporary")
    collect(jobs, (lambda f: {"job_id": "ext-a", "status": "queued"}, transient))
    job = next(iter(jobs.values()))
    assert job["status"] == "queued" and job["poll_errors"] == 1
    class FatalError(Exception): fatal = True
    def fatal(_): raise FatalError("missing")
    collect(jobs, (lambda f: {"job_id": "ext-a", "status": "queued"}, fatal))
    assert job["status"] == "failed"
