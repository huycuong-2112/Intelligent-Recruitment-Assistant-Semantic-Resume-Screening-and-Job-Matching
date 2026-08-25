from src.Matching.experience_matcher import match_experience

def test_explicit_zero_is_not_unknown():
    jd = {"experience": {"minimum_years": 1.0}, "responsibilities": ["Build APIs"]}
    artifact = {"experience_chunks": [{"chunk_id": "p1", "text": "Build APIs", "source_type": "project", "vector": [1.0, 0.0]}]}
    jd_artifact = {"responsibility_vectors": {"Build APIs": [1.0, 0.0]}}
    zero = match_experience(jd, {"experience": {"professional_years": 0.0}}, artifact, jd_artifact)
    unknown = match_experience(jd, {"experience": {"professional_years": None}}, artifact, jd_artifact)
    assert zero["years"]["score"] == 0.0 and zero["coverage"] == 1.0
    assert unknown["years"]["score"] is None and unknown["coverage"] == .7

def test_minimum_zero_is_satisfied():
    result = match_experience({"experience": {"minimum_years": 0}, "responsibilities": []}, {"experience": {"professional_years": None}})
    assert result["years"]["score"] == 1.0

def test_canonical_responsibility_chunks_are_consumed():
    jd = {"experience": {"minimum_years": 0.0}, "responsibilities": ["Build APIs"]}
    cv = {"experience": {"professional_years": 0.0}}
    cv_artifact = {"experience_chunks": [{"chunk_id": "p1", "text": "Build APIs", "source_type": "work", "vector": [1.0, 0.0]}]}
    jd_artifact = {"model": {"name": "m", "dimension": 2}, "responsibility_chunks": [{"text": "Build APIs", "vector": [1.0, 0.0]}]}
    result = match_experience(jd, cv, cv_artifact, jd_artifact)
    assert result["evidence"]["score"] > 0 and result["score"] > 0.3 and result["coverage"] == 1.0

def test_continuous_subthreshold_similarity_is_not_zeroed():
    jd = {"experience": {"minimum_years": 0.0}, "responsibilities": ["Build APIs"]}
    cv = {"experience": {"professional_years": 0.0}}
    result = match_experience(jd, cv, {"experience_chunks": [{"chunk_id": "p1", "vector": [0.45, (1 - 0.45**2) ** 0.5], "text": "evidence"}]}, {"responsibility_chunks": [{"text": "Build APIs", "vector": [1.0, 0.0]}]})
    assert result["evidence"]["score"] == 0.45
    assert result["evidence"]["details"]["responsibilities"][0]["status"] == "low_match"
