import pytest
from src.Matching.semantic_matcher import match_semantic

def artifact(vector, name="m", dimension=2):
    return {"model": {"name": name, "dimension": dimension}, "profile": {"vector": vector}}

def test_semantic_bounds_missing_and_incompatible():
    assert match_semantic(artifact([1, 0]), artifact([1, 0]))["score"] == 1.0
    assert match_semantic({}, artifact([1, 0]))["score"] is None
    with pytest.raises(ValueError): match_semantic(artifact([1, 0]), artifact([1, 0], name="other"))
