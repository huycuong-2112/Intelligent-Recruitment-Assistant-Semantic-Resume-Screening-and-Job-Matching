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
