from src.Normalization.jd_normalizer import normalize_jd


def test_jd_fields_remain_separate():
    result = normalize_jd({"id": "jd1", "parsed_data": {"job_title": "AI Engineer", "required_skills": ["RESTful APIs"], "preferred_skills": ["LLMs"], "min_experience_years": 2, "responsibilities": ["Build models"]}})
    assert result["skills"] == {"required": ["REST API"], "preferred": ["LLM"]}
    assert result["experience"]["minimum_years"] == 2.0
    assert result["responsibilities"] == ["Build models"]


def test_jd_experience_distinguishes_zero_unknown_and_invalid():
    assert normalize_jd({"parsed_data": {"min_experience_years": "0 years"}})["experience"]["minimum_years"] == 0.0
    assert normalize_jd({"parsed_data": {}})["experience"]["minimum_years"] is None
    assert normalize_jd({"parsed_data": {"min_experience_years": "N/A"}})["experience"]["minimum_years"] is None
    assert normalize_jd({"parsed_data": {}}, domain="IT")["domain"] == "IT"
