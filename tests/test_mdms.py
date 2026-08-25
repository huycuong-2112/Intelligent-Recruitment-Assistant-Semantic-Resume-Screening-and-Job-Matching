import pytest
from src.Matching.mdms import aggregate_mdms

W = {"skill": .4, "experience": .3, "education": .1, "semantic": .2}

def test_mdms_arithmetic_and_policies():
    results = {name: {"score": score, "coverage": 1.0, "availability": "available", "status": "evaluated"} for name, score in {"skill": 1, "experience": 0, "education": 1, "semantic": .5}.items()}
    assert aggregate_mdms(results, W)["final_score"] == pytest.approx(.6)
    results["education"] = {"score": None, "coverage": 0, "availability": "unknown", "status": "source_unavailable"}
    assert aggregate_mdms(results, W)["status"] == "insufficient_data"
    results["education"] = {"score": None, "coverage": 0, "availability": "not_applicable", "status": "not_required"}
    assert aggregate_mdms(results, W)["final_score"] is not None
    with pytest.raises(ValueError): aggregate_mdms(results, {**W, "skill": -.1})
