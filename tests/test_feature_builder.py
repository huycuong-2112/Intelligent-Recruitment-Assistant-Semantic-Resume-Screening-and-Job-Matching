from src.Representation.feature_builder import build_cv_features, build_jd_features


def test_feature_boundary_preserves_provenance_missingness_and_domain():
    cv = build_cv_features({"id": "cv1", "domain": "IT", "skills": {"all": ["Python"], "explicit": ["Python"], "project_derived": []}, "experience": {"professional_years": None}})
    assert cv.id == "cv1" and cv.domain == "IT"
    assert cv.skill_provenance["explicit"] == ["Python"]
    assert cv.professional_years is None
    cv_zero = build_cv_features({"experience": {"professional_years": 0.0}})
    assert cv_zero.professional_years == 0.0
    jd = build_jd_features({"domain": "IT", "skills": {"required": ["Python"], "preferred": ["Docker"]}, "responsibilities": ["Build APIs"]})
    assert jd.required_skills == ["Python"] and jd.preferred_skills == ["Docker"]
    assert jd.responsibilities == ["Build APIs"] and jd.domain == "IT"
