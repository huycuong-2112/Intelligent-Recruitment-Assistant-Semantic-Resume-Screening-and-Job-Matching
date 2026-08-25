import pytest
from src.Evaluation.ground_truth import validate_ground_truth

def valid():
    return {"jd_id":"jd1","domain":"IT","candidates":[{"cv_id":"cv1","relevance_grade":3,"ground_truth_score":90}]}

def test_valid_and_rejections():
    assert validate_ground_truth(valid()).candidates[0].cv_id == "cv1"
    for change in ({"cv_id":""},{"relevance_grade":4},{"ground_truth_score":101}):
        data=valid(); data["candidates"][0].update(change)
        with pytest.raises(ValueError): validate_ground_truth(data)
    data=valid(); data["candidates"].append(data["candidates"][0].copy())
    with pytest.raises(ValueError): validate_ground_truth(data)
