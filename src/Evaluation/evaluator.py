from __future__ import annotations
from typing import Any
from ground_truth import GroundTruthRecord
from metrics import mean_absolute_error, ndcg_at_k, recall_at_k, spearman_rank_correlation

def evaluate(ground_truth: GroundTruthRecord, system_results: list[dict[str, Any]], method: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    gt_ids = [candidate.cv_id for candidate in ground_truth.candidates]; result_ids = [item.get("cv_id") for item in system_results]
    if set(gt_ids) != set(result_ids) or len(gt_ids) != len(result_ids): raise ValueError("ground-truth and system candidate IDs must align exactly")
    if any(item.get("score_0_1", item.get("final_score")) is None for item in system_results): raise ValueError("evaluation incomplete: a system score is None")
    by_id = {item["cv_id"]: item for item in system_results}; ordered = [by_id[candidate.cv_id] for candidate in ground_truth.candidates]
    gt_grades = [candidate.overall for candidate in ground_truth.candidates]; gt_scores = gt_grades; predictions = [item.get("score_0_1", item.get("final_score")) for item in ordered]
    warnings: list[str] = []; ks = (config or {}).get("evaluation", {}).get("k_values", [5, 10, 15]); threshold = (config or {}).get("relevance", {}).get("relevant_threshold", 2)
    ranked_grades = [grade for _, grade in sorted(zip(predictions, gt_grades), key=lambda pair: pair[0], reverse=True)]
    metrics = {f"recall@{k}": recall_at_k(ranked_grades, k=k, threshold=threshold, warnings=warnings) for k in ks}; metrics.update({f"ndcg@{k}": ndcg_at_k(ranked_grades, k=k, warnings=warnings) for k in ks}); metrics["spearman"] = spearman_rank_correlation(predictions, gt_scores, warnings); metrics["mae"] = mean_absolute_error(gt_scores, predictions)
    return {"jd_id": ground_truth.jd_id, "method": method, "n_candidates": len(gt_ids), "metrics": metrics, "warnings": warnings}
