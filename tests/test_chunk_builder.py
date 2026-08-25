from src.Representation.chunk_builder import build_cv_chunks, build_jd_chunks, format_chunk_debug


def test_structure_aware_cv_chunks_and_impact():
    chunks = build_cv_chunks({"id": "cv_011", "projects": [{"name": "Interview Copilot", "description": "Designed an API Gateway with FastAPI and Redis Queue for asynchronous event handling, engineered a custom Rule Engine to enforce security guardrails, and built a Streamlit dashboard for real-time monitoring.", "impact_metrics": ["97.33% classification accuracy"]}]})
    assert chunks and all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.source_type == "project" for chunk in chunks)
    assert {chunk.source_name for chunk in chunks} == {"Interview Copilot"}
    assert any("97.33%" in chunk.text for chunk in chunks)
    assert not any(chunk.text.lower() in {"and built", "for real-time"} for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in build_cv_chunks({"id": "cv_011", "projects": [{"name": "Interview Copilot", "description": "Designed an API Gateway with FastAPI and Redis Queue for asynchronous event handling, engineered a custom Rule Engine to enforce security guardrails, and built a Streamlit dashboard for real-time monitoring.", "impact_metrics": ["97.33% classification accuracy"]}]})]
    assert "cv_011" in format_chunk_debug(chunks)


def test_jd_responsibilities_are_independent():
    chunks = build_jd_chunks({"id": "jd1", "role": {"job_title": "AI Engineer", "overview": "Build AI systems."}, "responsibilities": ["Prototype LLM features.", "Deploy services to production."]})
    responsibilities = [chunk for chunk in chunks if chunk.source_type == "responsibility"]
    assert len(responsibilities) == 2
    assert all(chunk.document_id == "jd1" and chunk.text for chunk in responsibilities)

def test_real_work_responsibilities_and_impact_field_is_chunked_once():
    value = {"id": "cv_003", "experience": {"work_evidence": [{"company": "TEKO", "role": "Software Engineer Intern", "responsibilities_and_impact": ["Build robust APIs.", "Worked with teams."]}]}}
    chunks = build_cv_chunks(value)
    assert len(chunks) == 2
    assert any("Build robust APIs." in chunk.text for chunk in chunks)
    assert all(chunk.source_type == "work" and chunk.source_name == "TEKO" for chunk in chunks)
    assert len({chunk.text for chunk in chunks}) == len(chunks)
