from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WEIGHTS = ["skill", "experience", "education", "semantic"]
METRICS = ["ndcg@5", "ndcg@10", "ndcg@15", "spearman", "mae_0_to_3", "recall@5"]
SELECTED = {"skill": 0.4, "experience": 0.2, "education": 0.1, "semantic": 0.3}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "Data/Results/IT/GridSearch/jd_001/grid_search_results.csv"
    out = root / "Data/Results/IT/WeightSensitivity/jd_001"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    cols = [f"w_{x}" for x in WEIGHTS]
    assert len(df) == 286 and df[cols].ge(0).all().all()
    assert ((df[cols].sum(axis=1) - 1).abs() < 1e-9).all()
    assert df[cols].drop_duplicates().shape[0] == len(df)
    assert all(round(float(v) * 10) == float(v) * 10 for v in df[cols].to_numpy().ravel())
    assert set(["ndcg@5", "ndcg@10", "ndcg@15", "spearman", "mae_0_to_3", "recall@5"]).issubset(df.columns)

    dist = []
    for m in METRICS:
        s = df[m]
        dist.append({"metric": m, "min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(), "std": s.std(ddof=0)})
    pd.DataFrame(dist).to_csv(out / "global_metric_distribution.csv", index=False)

    best = float(df["ndcg@10"].max())
    regions = []
    for label, tol in [("within_0.01", 0.01), ("within_0.02", 0.02)]:
        sub = df[df["ndcg@10"] >= best - tol].copy()
        row = {"region": label, "count": len(sub)}
        for w in WEIGHTS:
            row[f"{w}_min"] = sub[f"w_{w}"].min(); row[f"{w}_max"] = sub[f"w_{w}"].max(); row[f"{w}_mean"] = sub[f"w_{w}"].mean(); row[f"{w}_median"] = sub[f"w_{w}"].median(); row[f"{w}_std"] = sub[f"w_{w}"].std(ddof=0)
        regions.append(row)
    pd.DataFrame(regions).to_csv(out / "near_best_weight_ranges.csv", index=False)

    sensitivity = {}
    for w in WEIGHTS:
        rows = []
        for value in [i / 10 for i in range(11)]:
            sub = df[df[f"w_{w}"] == value]
            rows.append({"weight_value": value, "configuration_count": len(sub), "best_ndcg10": sub["ndcg@10"].max(), "mean_ndcg10": sub["ndcg@10"].mean(), "std_ndcg10": sub["ndcg@10"].std(ddof=0), "best_ndcg5": sub["ndcg@5"].max(), "mean_spearman": sub["spearman"].mean(), "mean_mae": sub["mae_0_to_3"].mean(), "best_spearman": sub["spearman"].max(), "best_mae": sub["mae_0_to_3"].min()})
        sensitivity[w] = pd.DataFrame(rows)
        sensitivity[w].to_csv(out / f"weight_sensitivity_{w}.csv", index=False)

    conditional = []
    for w in WEIGHTS:
        for value in [i / 10 for i in range(11)]:
            sub = df[df[f"w_{w}"] == value].sort_values(["ndcg@10", "spearman", "ndcg@5", "mae_0_to_3"], ascending=[False, False, False, True]).iloc[0]
            conditional.append({"component": w, "weight_value": value, "best_ndcg10": sub["ndcg@10"], "best_spearman": sub["spearman"], "corresponding_weights": "/".join(f"{sub[f'w_{x}']:.1f}" for x in WEIGHTS), "best_ndcg5": sub["ndcg@5"], "mae_0_to_3": sub["mae_0_to_3"]})
    pd.DataFrame(conditional).to_csv(out / "conditional_best_by_weight.csv", index=False)

    zero = []
    for w in WEIGHTS:
        sub = df[df[f"w_{w}"] == 0]
        zero.append({"component": w, "best_ndcg10": sub["ndcg@10"].max(), "within_0.01": bool((sub["ndcg@10"] >= best - .01).any()), "within_0.02": bool((sub["ndcg@10"] >= best - .02).any()), "count_within_0.01": int((sub["ndcg@10"] >= best - .01).sum()), "count_within_0.02": int((sub["ndcg@10"] >= best - .02).sum())})
    pd.DataFrame(zero).to_csv(out / "weight_zero_analysis.csv", index=False)

    for w in WEIGHTS:
        df[f"delta_{w}"] = df[f"w_{w}"] - SELECTED[w]
    df["l1_distance"] = sum(df[f"delta_{w}"].abs() for w in WEIGHTS)
    local = df.sort_values(["l1_distance", "objective_rank"]).head(20).copy()
    local.to_csv(out / "local_neighborhood.csv", index=False)
    perturb = df[(df.l1_distance > 0) & (df.l1_distance <= 0.2)].sort_values(["l1_distance", "objective_rank"]).copy()
    perturb.to_csv(out / "one_step_perturbations.csv", index=False)

    top10 = df.nsmallest(10, "objective_rank")
    variance = []
    for label, sub in [("top_10", top10), ("within_0.01", df[df["ndcg@10"] >= best - .01]), ("within_0.02", df[df["ndcg@10"] >= best - .02])]:
        for w in WEIGHTS:
            variance.append({"region": label, "component": w, "count": len(sub), "mean": sub[f"w_{w}"].mean(), "variance": sub[f"w_{w}"].var(ddof=0), "std": sub[f"w_{w}"].std(ddof=0), "min": sub[f"w_{w}"].min(), "max": sub[f"w_{w}"].max()})
    pd.DataFrame(variance).to_csv(out / "weight_variance_summary.csv", index=False)

    selected_row = df[(df[cols] == pd.Series({f"w_{w}": SELECTED[w] for w in WEIGHTS})).all(axis=1)].iloc[0]
    report = {"source": str(source), "development_n": 18, "grid_configurations": len(df), "best_ndcg10": best, "winner": selected_row.to_dict(), "global_distribution": dist, "near_best": regions, "zero_weight": zero, "closest_alternatives": local.head(10).to_dict("records"), "one_step_count": len(perturb)}
    (out / "weight_sensitivity_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
