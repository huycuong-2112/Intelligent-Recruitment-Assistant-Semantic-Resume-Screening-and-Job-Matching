from app.frontend.utils import extraction_jobs
from app.frontend.utils.extraction_jobs import collect, consume_completed_jobs, start_jobs, submit_next

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

def test_mixed_file_states_remain_independent():
    jobs, _ = start_jobs([File("slow.jpg"), File("ok.pdf")], lambda f: {"job_id": "ext-" + f.name, "status": "running"})
    first = next(iter(jobs.values()))
    first.update(job_id="ext-slow.jpg", status="running", submitted_at=0)
    second = list(jobs.values())[1]
    second.update(job_id="ext-ok.pdf", status="running", submitted_at=0)
    def status(job_id):
        return {"status": "running"} if job_id == "ext-slow.jpg" else {"status": "completed", "result": {"filename": "ok.pdf"}}
    collect(jobs, (lambda f: {"job_id": "unused", "status": "queued"}, status))
    assert first["status"] in {"running", "long_running"}
    assert second["status"] == "completed" and second["result"]["filename"] == "ok.pdf"

def test_completed_result_is_persisted_as_refinable_before_queue_cleanup():
    jobs = {"a": {"filename": "a.jpg", "order": 0, "status": "completed", "result": {"filename": "a.jpg", "run_id": "r", "document_id": "cv_1", "parsed": {}, "ui_features": []}}}
    candidates, changed = consume_completed_jobs(jobs, [])
    assert changed and candidates[0]["refinable"] is True and candidates[0]["document_id"] == "cv_1"

def test_completed_handoff_survives_rerun_and_failed_has_no_refine():
    jobs = {"a": {"filename": "a.jpg", "order": 0, "status": "completed", "result": {"filename": "a.jpg", "run_id": "r", "document_id": "cv_1"}}, "b": {"filename": "b.pdf", "order": 1, "status": "failed", "error": "bad"}}
    first, _ = consume_completed_jobs(jobs, [])
    second, changed = consume_completed_jobs({}, first)
    assert first == second and changed is False and first[0]["refinable"] is True and all(c.get("refinable") for c in first)

def test_two_completed_jobs_produce_two_refinable_candidates():
    jobs={"a":{"filename":"cv_018.pdf","order":0,"status":"completed","result":{"filename":"cv_018.pdf","document_id":"cv_a"}},
          "b":{"filename":"003.jpg","order":1,"status":"completed","result":{"filename":"003.jpg","document_id":"cv_b"}}}
    candidates, changed=consume_completed_jobs(jobs, [])
    assert changed and [c["document_id"] for c in candidates] == ["cv_a","cv_b"]
    assert all(c["refinable"] for c in candidates)

def test_completed_first_preserves_pending_second():
    jobs={"a":{"filename":"a.pdf","order":0,"status":"completed","result":{"filename":"a.pdf","document_id":"cv_a"}},
          "b":{"filename":"b.jpg","order":1,"status":"pending"}}
    candidates, _=consume_completed_jobs(jobs, [])
    assert [c["document_id"] for c in candidates] == ["cv_a"] and candidates[0]["refinable"]
    assert jobs["b"]["status"] == "pending"

def test_second_later_completion_adds_without_overwriting_first():
    first={"filename":"a.pdf","document_id":"cv_a","refinable":True}
    jobs={"b":{"filename":"b.jpg","order":1,"status":"completed","result":{"filename":"b.jpg","document_id":"cv_b"}}}
    candidates, changed=consume_completed_jobs(jobs, [first])
    assert changed and {c["document_id"] for c in candidates} == {"cv_a","cv_b"}

def test_same_filename_updates_candidate_without_duplicate():
    existing={"filename":"a.pdf","document_id":"cv_a","refinable":False}
    jobs={"a":{"filename":"a.pdf","status":"completed","result":{"filename":"a.pdf","document_id":"cv_a"}}}
    candidates, _=consume_completed_jobs(jobs,[existing])
    assert len(candidates)==1 and candidates[0]["refinable"] is True

def test_mixed_failed_and_successful_keeps_successful_refinable():
    jobs={"bad":{"filename":"bad.pdf","status":"failed"},"ok":{"filename":"ok.jpg","order":1,"status":"completed","result":{"filename":"ok.jpg","document_id":"cv_ok"}}}
    candidates, _=consume_completed_jobs(jobs,[])
    assert len(candidates)==1 and candidates[0]["document_id"]=="cv_ok" and candidates[0]["refinable"]
