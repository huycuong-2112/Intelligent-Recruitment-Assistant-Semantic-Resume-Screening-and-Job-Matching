from app.api.services.teammate_explanation_adapter import generate_teammate_narrative

def _xai(score=.72):
    return {"cv_id":"cv_demo","jd_id":"jd_demo","job_title":"AI Engineer Intern",
            "decision":{"final_score":score},
            "dimensions":{"skill":{"matched_required":["Python"],"missing_required":["Git"]},
                          "experience":{"years":{"candidate_years":1,"minimum_years":0}},
                          "education":{"degree":{"candidate":"Bachelor","status":"satisfied"},"field":{"candidate":"Computer Science","status":"matched"}}}}

def test_offline_adapter_returns_canonical_narrative():
    narrative, method, model, fallback = generate_teammate_narrative(_xai(), {"facts":{},"interview_topics":[]}, "offline")
    assert method == "offline_deterministic" and model is None and not fallback
    assert set(("summary", "strengths", "gaps", "interview_focus", "disclaimer")) <= narrative.keys()

def test_factory_groq_path_uses_teammate_shape(monkeypatch):
    class Msg: content = '{"fit_level":"Cao","summary":"Tốt.","primary_factor":"skills","strengths":["Python"],"gaps":[],"cv_improvements":[]}'
    class Completions:
        def create(self, **kwargs): return type("R", (), {"choices":[type("C", (), {"message":Msg()})()]})()
    class Client: chat = type("Chat", (), {"completions":Completions()})()
    narrative, method, model, fallback = generate_teammate_narrative(_xai(), {"facts":{},"interview_topics":[]}, "auto", lambda: Client())
    assert method == "groq_llm" and model == "teammate_candidate_feedback" and not fallback
