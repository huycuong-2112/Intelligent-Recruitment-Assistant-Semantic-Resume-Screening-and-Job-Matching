import json

from src.Data_loader import jd_parser
from app.api.adapters.presentation_adapter import parsed_jd_to_ui_features
from app.api.adapters.confirm_override import build_confirm_override
from src.Normalization.jd_normalizer import normalize_jd
import pytest


def test_parse_cleaned_jds_attempts_groq_in_auto_mode(tmp_path, monkeypatch):
    source = tmp_path / "cleaned.json"
    output = tmp_path / "parsed.json"
    source.write_text(json.dumps([{"id": "jd_auto", "filename": "x.pdf", "content": "Job description text"}]))

    class FakeGroq:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr(jd_parser, "Groq", FakeGroq)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(jd_parser, "parse_jd_llm", lambda text, client: jd_parser.StructuredJobDescription(job_title="Role"))
    result = jd_parser.parse_cleaned_jds(source, output, offline=False)
    assert result[0]["extraction_method"] == "groq_llm"


def test_parse_cleaned_jds_offline_skips_groq(tmp_path, monkeypatch):
    source = tmp_path / "cleaned.json"
    output = tmp_path / "parsed.json"
    source.write_text(json.dumps([{"id": "jd_offline", "filename": "x.pdf", "content": "Job description text"}]))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(jd_parser, "parse_jd_llm", lambda *_: (_ for _ in ()).throw(AssertionError("must not call Groq")))
    result = jd_parser.parse_cleaned_jds(source, output, offline=True)
    assert result[0]["extraction_method"] == "offline_hybrid"


def test_jd_presentation_ids_unique_across_education_sources():
    record = {"id": "jd_1", "parsed_data": {"required_degree": "Bachelor", "preferred_fields": ["Computer Science"]}}
    features = parsed_jd_to_ui_features(record)
    assert len(features) == len({f["id"] for f in features})


def test_jd_manual_required_skill_duplicate_is_rejected():
    record = {"id": "jd_1", "parsed_data": {"required_skills": ["Python programming"]}}
    features = parsed_jd_to_ui_features(record)
    with pytest.raises(ValueError):
        build_confirm_override("jd_1", features, features + [{"name": " python programming ", "category": "Required Skills", "feature_type": "required_skill", "source_type": "manual_ui"}])


def test_jd_generic_field_qualifiers_are_not_atomic():
    for values, expected in [(["Computer Science", "Related Field"], ["Computer Science"]), (["Accounting", "Finance", "related disciplines"], ["Accounting", "Finance"]), (["Marketing"], ["Marketing"]), (["Nursing"], ["Nursing"]), (["Mechanical Engineering", "equivalent field"], ["Mechanical Engineering"])]:
        out = normalize_jd({"id": "jd", "parsed_data": {"preferred_fields": values}})
        assert out["education"]["preferred_fields"] == expected
