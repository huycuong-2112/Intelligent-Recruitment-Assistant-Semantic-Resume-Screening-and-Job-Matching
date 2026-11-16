import json
from pathlib import Path

from src.Explainability.evidence_builder import build_xai


ROOT = Path(__file__).resolve().parents[1]


def inputs():
    jd = json.loads((ROOT / "Data/Normalized/IT/JD/jd_001.json").read_text(encoding="utf-8"))
    cv = json.loads((ROOT / "Data/Normalized/IT/CV/cv_001.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "Data/Results/IT/Matching/jd_001/cv_001.json").read_text(encoding="utf-8"))
    frozen = {"skill": .4, "experience": .2, "education": .1, "semantic": .3}
    return jd, cv, result, frozen


def test_builder_preserves_score_components_and_refs():
    jd, cv, result, frozen = inputs(); x = build_xai(jd, cv, result, result["mdms"], frozen)
    assert x.decision.final_score == result["mdms"]["final_score"]
    assert x.dimensions["skill"].score == result["skill"]["score"]
    assert x.dimensions["experience"].score == result["experience"]["score"]
    refs = [i.get("evidence_ref") for i in x.strength_candidates + x.gap_candidates + x.interview_focus]
    assert all(r is None or r in x.evidence_registry for r in refs)
    assert "ev_exp_cv_001_work_001" in x.evidence_registry
    assert x.evidence_registry["ev_exp_cv_001_work_001"].source_text == result["experience"]["evidence"]["details"]["responsibilities"][0]["matched_text"]


def test_builder_is_deterministic_and_has_no_vectors():
    jd, cv, result, frozen = inputs(); a = build_xai(jd, cv, result, result["mdms"], frozen).model_dump(mode="json"); b = build_xai(jd, cv, result, result["mdms"], frozen).model_dump(mode="json")
    assert a == b
    assert "vector" not in json.dumps(a).lower()
    assert not any(i.get("type", "").startswith("semantic") for i in a["strength_candidates"])


def test_preferred_missing_skills_are_not_gaps():
    jd, cv, result, frozen = inputs(); x = build_xai(jd, cv, result, result["mdms"], frozen)
    assert all(i.get("type") != "missing_preferred_skill" for i in x.gap_candidates)


def test_weights_contributions_and_final_score_are_consistent():
    jd, cv, result, frozen = inputs(); x = build_xai(jd, cv, result, result["mdms"], frozen)
    contributions = [d.weighted_contribution for d in x.dimensions.values() if d.weighted_contribution is not None]
    assert abs(sum(contributions) - x.decision.final_score) < 1e-12
    for name, dimension in x.dimensions.items():
        assert dimension.weight == x.decision.effective_weights[name]
        assert dimension.weight == x.decision.weights[name]


def test_model_version_matches_source_mdms_weights():
    jd, cv, result, frozen = inputs(); x = build_xai(jd, cv, result, result["mdms"], frozen)
    assert x.decision.model_version == "mdms_equal_v1"
    assert x.decision.weights == {"skill": .25, "experience": .25, "education": .25, "semantic": .25}
