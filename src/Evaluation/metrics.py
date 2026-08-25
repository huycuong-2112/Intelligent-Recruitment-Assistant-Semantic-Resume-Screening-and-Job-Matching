from __future__ import annotations
import math
from statistics import mean
from typing import Any, Sequence

def _warn(warnings: list[str] | None, message: str) -> None:
    if warnings is not None: warnings.append(message)

def recall_at_k(relevance: Sequence[float], ranked_ids: Sequence[str] | None = None, k: int = 10, threshold: int = 2, warnings: list[str] | None = None) -> float | None:
    if k > len(relevance): _warn(warnings, f"Requested K={k} but only N={len(relevance)} candidates are available."); return None
    relevant_total = sum(value >= threshold for value in relevance)
    if relevant_total == 0: _warn(warnings, "Recall is undefined because there are no relevant candidates."); return None
    return sum(value >= threshold for value in relevance[:k]) / relevant_total

def ndcg_at_k(relevance: Sequence[float], ranked_ids: Sequence[str] | None = None, k: int = 10, warnings: list[str] | None = None) -> float | None:
    if k > len(relevance): _warn(warnings, f"Requested K={k} but only N={len(relevance)} candidates are available."); return None
    def dcg(values: Sequence[float]) -> float: return sum((2 ** value - 1) / math.log2(index + 2) for index, value in enumerate(values))
    actual, ideal = dcg(relevance[:k]), dcg(sorted(relevance, reverse=True)[:k])
    if ideal == 0: _warn(warnings, "nDCG is undefined because ideal DCG is zero."); return None
    return actual / ideal

def spearman_rank_correlation(system_scores: Sequence[float], ground_truth_scores: Sequence[float], warnings: list[str] | None = None) -> float | None:
    if len(system_scores) != len(ground_truth_scores) or len(system_scores) < 2: _warn(warnings, "Spearman is undefined for fewer than two aligned candidates."); return None
    if len(set(system_scores)) == 1 or len(set(ground_truth_scores)) == 1: _warn(warnings, "Spearman is undefined for constant values."); return None
    try:
        from scipy.stats import spearmanr
        value = float(spearmanr(system_scores, ground_truth_scores).statistic)
    except Exception: return None
    return value if math.isfinite(value) else None

def mean_absolute_error(ground_truth_scores: Sequence[float], system_scores: Sequence[float], warnings: list[str] | None = None) -> float:
    if len(ground_truth_scores) != len(system_scores): raise ValueError("MAE inputs must be aligned and equal length")
    predictions = [float(value) * 3 if 0 <= float(value) <= 1 else float(value) for value in system_scores]
    if any(value < 0 or value > 3 for value in predictions): raise ValueError("system predictions must be in [0,1] or [0,3]")
    if any(value < 0 or value > 3 for value in ground_truth_scores): raise ValueError("ground truth scores must be in [0,3]")
    return mean(abs(float(actual) - prediction) for actual, prediction in zip(ground_truth_scores, predictions)) if predictions else 0.0

# Backward-compatible aliases used by prior evaluation code.
compute_ndcg_at_k = ndcg_at_k
