import pytest
from src.Evaluation.metrics import recall_at_k, ndcg_at_k, spearman_rank_correlation, mean_absolute_error

def test_metrics():
    assert recall_at_k([3,0,2,0,2,2], k=5, threshold=2) == .75
    assert ndcg_at_k([3,2,1], k=3) == pytest.approx(1)
    assert ndcg_at_k([0,0], k=2) is None
    assert spearman_rank_correlation([3,2,1],[1,2,3]) == pytest.approx(-1)
    assert spearman_rank_correlation([1,1],[1,2]) is None
    assert mean_absolute_error([3,2,0],[.8,.75,.35]) == pytest.approx((.6+.25+1.05)/3)
    warnings=[]
    assert recall_at_k([3]*10,k=15,warnings=warnings) is None and warnings
