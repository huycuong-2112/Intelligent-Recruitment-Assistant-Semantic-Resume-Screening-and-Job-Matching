from __future__ import annotations
from itertools import product
from typing import Any, Iterable
from src.Evaluation.metrics import mean_absolute_error, ndcg_at_k, recall_at_k, spearman_rank_correlation

DIMENSIONS = ("skill", "experience", "education", "semantic")

def simplex_grid(step_units: int = 10) -> list[tuple[float, float, float, float]]:
    return [(a/step_units,b/step_units,c/step_units,(step_units-a-b-c)/step_units) for a,b,c in product(range(step_units+1), repeat=3) if a+b+c <= step_units]

def _metrics(scores: list[float], gt: list[int], threshold: int = 2) -> dict[str, float | None]:
    ranked = [grade for _,grade in sorted(zip(scores,gt), key=lambda pair: pair[0], reverse=True)]
    warnings: list[str] = []
    result={f"recall@{k}":recall_at_k(ranked,k=k,threshold=threshold,warnings=warnings) for k in (5,10,15)}
    result.update({f"ndcg@{k}":ndcg_at_k(ranked,k=k,warnings=warnings) for k in (5,10,15)})
    result["spearman"]=spearman_rank_correlation(scores,gt,warnings); result["mae_0_to_3"]=mean_absolute_error(gt,scores); return result

def search(rows: list[dict[str,Any]], threshold: int = 2, tolerance: float = 1e-9) -> tuple[list[dict[str,Any]], dict[str,Any]]:
    evaluated=[]
    for weights in simplex_grid():
        scores=[sum(weights[i]*float(row[f"{DIMENSIONS[i]}_score"]) for i in range(4)) for row in rows]; metrics=_metrics(scores,[int(row["gt_overall"]) for row in rows],threshold)
        evaluated.append({**dict(zip(("w_skill","w_experience","w_education","w_semantic"),weights)),**metrics})
    def key(item):
        return (-float(item["ndcg@10"] or -1),-float(item["spearman"] or -1),float(item["ndcg@5"] or -1),float(item["mae_0_to_3"]),sum(abs(item[name]-.25) for name in ("w_skill","w_experience","w_education","w_semantic")),tuple(item[name] for name in ("w_skill","w_experience","w_education","w_semantic")))
    ordered=sorted(evaluated,key=key); best=ordered[0]; best["objective_rank"]=1
    for rank,item in enumerate(ordered,1): item["objective_rank"]=rank
    return ordered,best
