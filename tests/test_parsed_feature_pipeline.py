from copy import deepcopy

from app.api.adapters.presentation_adapter import parsed_cv_to_ui_features
from app.api.adapters.confirm_override import build_confirm_override, apply_cv_override
import pytest
from src.Normalization.cv_normalizer import normalize_cv
from src.Representation.feature_builder import build_cv_features
from app.frontend.utils.manual_feature_contract import types_for, subtype_selector_visible


def _parsed():
    return {
        "id": "cv_pipeline",
        "domain": "IT",
        "parsed_data": {
            "summary": "profile",
            "skills": ["Python"],
            "education_degree": "Bachelor",
            "education_field": "Computer Science",
            "education_history": [{"degree": "Bachelor", "institution": "U"}],
            "experience_years": 2,
            "job_titles": ["Engineer"],
            "work_experience": [{"role": "Engineer", "responsibilities_and_impact": ["Build systems"]}],
            "projects": [{"name": "Demo", "technologies": ["Docker"]}],
            "certifications": ["Cert"],
        },
    }


def test_parsed_to_normalized_features_and_presentation_preserves_groups():
    parsed = _parsed()
    normalized = normalize_cv(parsed)
    features = build_cv_features(normalized)
    assert features.skills == ["Python", "Docker"]
    assert features.education["history"]
    presentation = parsed_cv_to_ui_features(parsed)
    assert {f["category"] for f in presentation} == {"Education", "Skills", "Experience", "Projects"}
    assert any(f["source_type"] == "project_technology" for f in presentation)


def test_projection_is_immutable_and_deterministic():
    parsed = _parsed()
    before = deepcopy(parsed)
    first = parsed_cv_to_ui_features(parsed)
    second = parsed_cv_to_ui_features(parsed)
    assert parsed == before
    assert first == second


def test_confirm_payload_has_unique_ids_and_no_edit_succeeds():
    features = parsed_cv_to_ui_features(_parsed())
    ids = [f["id"] for f in features]
    assert len(ids) == len(set(ids))
    override = build_confirm_override("cv_pipeline", features, features)
    assert len(override["kept_feature_ids"]) == len(features)
    assert override["removed_feature_ids"] == []


def test_manual_group_additions_and_skill_duplicate_rejection():
    parsed = _parsed()
    original = parsed_cv_to_ui_features(parsed)
    with pytest.raises(ValueError):
        build_confirm_override("cv_pipeline", original, original + [{"name": " python ", "category": "Skills", "source_type": "manual_ui"}])
    additions = [
        {"name": "Master", "category": "Education", "feature_type": "degree", "source_type": "manual_ui"},
        {"name": "Lead", "category": "Experience", "feature_type": "role", "source_type": "manual_ui"},
        {"name": "Portfolio", "category": "Projects", "feature_type": "project", "source_type": "manual_ui"},
    ]
    override = build_confirm_override("cv_pipeline", original, original + additions)
    result = apply_cv_override(parsed, override)
    assert len(result["unsupported_actions"]) == 0
    assert all(x["source_type"] == "manual_ui" for x in override["added_features"])


def test_manual_category_type_contract_is_server_validated():
    original = parsed_cv_to_ui_features(_parsed())
    with pytest.raises(ValueError, match="category/type"):
        build_confirm_override("cv_pipeline", original, original + [{"name": "x", "category": "Experience", "feature_type": "skill", "source_type": "manual_ui"}])


def test_manual_subtype_contract_is_category_safe():
    assert types_for("Skills") == ["skill"] and not subtype_selector_visible("Skills")
    assert types_for("Education") == ["degree", "field"] and subtype_selector_visible("Education")
    assert types_for("Experience") == ["role", "responsibility"] and subtype_selector_visible("Experience")
    assert types_for("Projects") == ["project_name", "project_evidence"] and subtype_selector_visible("Projects")
