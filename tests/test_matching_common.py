from src.Matching.common import ComponentResult, AVAILABLE, UNKNOWN, NOT_APPLICABLE, aggregate_components
import pytest

def test_local_aggregation_semantics():
    score, coverage = aggregate_components([ComponentResult(None, UNKNOWN, "unknown", .3), ComponentResult(.8, AVAILABLE, "matched", .7)])
    assert score == pytest.approx(.8) and coverage == pytest.approx(.7)
    score, coverage = aggregate_components([ComponentResult(0.0, AVAILABLE, "no_evidence", .3), ComponentResult(.8, AVAILABLE, "matched", .7)])
    assert abs(score - .56) < 1e-9 and coverage == 1.0
    assert aggregate_components([ComponentResult(None, UNKNOWN, "unknown", 1.0)]) == (None, 0.0)
    assert aggregate_components([ComponentResult(None, NOT_APPLICABLE, "not_required", 1.0), ComponentResult(1.0, AVAILABLE, "matched", 1.0)]) == (1.0, 1.0)
