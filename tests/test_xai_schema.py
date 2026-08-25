import pytest
from pydantic import ValidationError

from src.Explainability.schemas import EvidenceStatus, XAIOutput, compact_llm_payload


def sample():
    return {
        "jd_id": "jd_001", "cv_id": "cv_001",
        "decision": {"final_score": .5, "status": "evaluated", "weights": {"skill": .4, "experience": .2, "education": .1, "semantic": .3}},
        "dimensions": {k: {"score": .5, "status": "AVAILABLE", "weight": w} for k, w in {"skill": .4, "experience": .2, "education": .1, "semantic": .3}.items()},
        "strength_candidates": [{"type": "match", "evidence_ref": "ev1"}],
        "evidence_registry": {"ev1": {"evidence_id": "ev1", "source_type": "cv_skill"}},
    }


def test_required_fields_and_payload():
    obj = XAIOutput.model_validate(sample())
    assert obj.schema_version == "xai_v1"
    assert compact_llm_payload(obj)["decision"]["final_score"] == .5


def test_weights_must_sum_to_one():
    value = sample(); value["decision"]["weights"]["semantic"] = .2
    with pytest.raises(ValidationError): XAIOutput.model_validate(value)


def test_score_range_and_status_enum():
    value = sample(); value["dimensions"]["skill"]["score"] = 2
    with pytest.raises(ValidationError): XAIOutput.model_validate(value)
    assert EvidenceStatus.NO_EVIDENCE.value == "NO_EVIDENCE"


def test_unknown_evidence_reference_rejected():
    value = sample(); value["gap_candidates"] = [{"evidence_ref": "missing"}]
    with pytest.raises(ValidationError): XAIOutput.model_validate(value)

