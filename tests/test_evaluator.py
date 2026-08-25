import pytest
from src.Evaluation.evaluator import evaluate
from src.Evaluation.ground_truth import validate_ground_truth

def gt():
    return validate_ground_truth({"jd_id":"jd1","domain":"IT","candidates":[{"cv_id":"a","relevance_grade":3,"ground_truth_score":90},{"cv_id":"b","relevance_grade":0,"ground_truth_score":20}]})

def test_evaluator_alignment_none_and_metrics():
    result=[{"cv_id":"a","score_0_1":.8},{"cv_id":"b","score_0_1":.2}]
    out=evaluate(gt(),result,"mdms",{"evaluation":{"k_values":[5]},"relevance":{"relevant_threshold":2}})
    assert "recall@5" in out["metrics"] and out["metrics"]["recall@5"] is None
    with pytest.raises(ValueError): evaluate(gt(),[{"cv_id":"a","score_0_1":.8}],"mdms")
    with pytest.raises(ValueError): evaluate(gt(),[{"cv_id":"a","score_0_1":None},{"cv_id":"b","score_0_1":.2}],"mdms")
