from src.Normalization.cv_normalizer import normalize_cv


def test_cv_provenance_and_years():
    result = normalize_cv({"id": "cv1", "parsed_data": {"skills": ["Python", "Git"], "experience_years": 0.0, "projects": [{"technologies": ["Git", "LLMs"], "description": "Built a bot"}], "work_experience": [{"description": "Intern"}]}})
    assert result["skills"]["explicit"] == ["Python", "Git"]
    assert result["skills"]["project_derived"] == ["Git", "LLM"]
    assert result["skills"]["all"] == ["Python", "Git", "LLM"]
    assert result["experience"]["professional_years"] == 0.0
    assert result["experience"]["project_evidence"] == ["Built a bot"]


def test_cv_experience_distinguishes_zero_unknown_and_invalid():
    assert normalize_cv({"parsed_data": {"experience_years": 0}})["experience"]["professional_years"] == 0.0
    assert normalize_cv({"parsed_data": {}})["experience"]["professional_years"] is None
    assert normalize_cv({"parsed_data": {"experience_years": "unknown"}})["experience"]["professional_years"] is None
    assert normalize_cv({"parsed_data": {"experience_years": -1}})["experience"]["professional_years"] is None
    assert normalize_cv({"parsed_data": {}})["domain"] is None
    assert normalize_cv({"parsed_data": {}}, domain="IT")["domain"] == "IT"
