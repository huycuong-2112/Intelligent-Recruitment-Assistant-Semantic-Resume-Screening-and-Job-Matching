from src.Matching.skill_matcher import match_skills

def test_skill_exact_missing_and_weighting():
    jd = {"skills": {"required": ["REST API", "Docker"], "preferred": ["Git"]}}
    cv = {"skills": {"all": ["REST API", "Git"]}}
    result = match_skills(jd, cv)
    assert result["requirements"][0]["score"] == 1.0
    assert result["requirements"][1]["status"] == "no_evidence" and result["requirements"][1]["score"] == 0.0
    assert "Docker" in result["missing_required"] and "Git" in result["matched_preferred"]
