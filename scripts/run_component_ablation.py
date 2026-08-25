from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.Optimization.grid_search import _metrics


FULL = {"skill": 0.4, "experience": 0.2, "education": 0.1, "semantic": 0.3}


def simplex3(names: list[str]):
    for a in range(11):
        for b in range(11 - a):
            c = 10 - a - b
            yield {names[0]: a / 10, names[1]: b / 10, names[2]: c / 10}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "Data/Results/IT"
    out = base / "Ablation/Components/jd_001"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(base / "Evaluation/development/jd_001_candidate_diagnostics_continuous_experience.csv")
    assert len(d) == 18 and d.cv_id.is_unique
    gt = d.gt_overall.astype(int).tolist()
    components = list(FULL)

    def scores(weights):
        return [sum(weights[k] * row[f"{k}_score"] for k in weights) for _, row in d.iterrows()]

    def metrics(weights):
        return _metrics(scores(weights), gt)

    def ordering_key(item):
        return (
            -float(item["ndcg@10"] if item["ndcg@10"] is not None else -1),
            -float(item["spearman"] if item["spearman"] is not None else -1),
            -float(item["ndcg@5"] if item["ndcg@5"] is not None else -1),
            float(item["mae_0_to_3"]),
            sum(abs(item[n] - 1 / 3) for n in names),
            tuple(item[n] for n in names),
        )

    full_metrics = metrics(FULL)
    rows = [{"model": "full_tuned", **FULL, **full_metrics}]
    reoptimized = {}
    all_grids = []
    for removed in components:
        names = [n for n in components if n != removed]
        total = 1 - FULL[removed]
        frozen = {n: FULL[n] / total for n in names}
        rows.append({"model": f"no_{removed}_frozen_renorm", **{n: frozen.get(n, 0.0) for n in components}, **metrics(frozen)})
        if removed == "semantic":
            prior = json.loads((base / "Ablation/Semantic/jd_001/semantic_redundancy_report.json").read_text(encoding="utf-8"))
            best = {"removed": removed, **prior["no_semantic_best"]}
            reoptimized[removed] = {n: best[f"w_{n}"] for n in names}
            all_grids.extend(pd.read_csv(base / "Ablation/Semantic/jd_001/no_semantic_grid_search.csv").assign(removed=removed).to_dict("records"))
            rows.append({"model": f"no_{removed}_reoptimized", **{n: best.get(f"w_{n}", 0.0) for n in components}, **{k: best[k] for k in full_metrics}})
            continue
        grid = []
        for w in simplex3(names):
            m = metrics(w)
            grid.append({"removed": removed, **{n: w[n] for n in names}, **m})
        grid.sort(key=lambda x: (-float(x["ndcg@10"] or -1), -float(x["spearman"] or -1), -float(x["ndcg@5"] or -1), float(x["mae_0_to_3"]), sum(abs(x[n] - 1 / 3) for n in names), tuple(x[n] for n in names)))
        best = grid[0]
        reoptimized[removed] = {n: best[n] for n in names}
        all_grids.extend(grid)
        rows.append({"model": f"no_{removed}_reoptimized", **{n: best.get(n, 0.0) for n in components}, **{k: best[k] for k in full_metrics}})

    # Batch 5.1 used the same development data and objective; the values below are recomputed only
    # as a consistency check, while the persisted 3-component grid is the ablation source of truth.
    pd.DataFrame(all_grids).to_csv(out / "reoptimized_grids.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "full_ablation_metrics.csv", index=False)
    weight_rows = [{"removed": r, **reoptimized[r]} for r in components]
    pd.DataFrame(weight_rows).to_csv(out / "reoptimized_weights.csv", index=False)

    deltas = []
    for removed in components:
        m = next(x for x in rows if x["model"] == f"no_{removed}_reoptimized")
        deltas.append({"removed": removed, **{k: full_metrics[k] - m[k] for k in full_metrics}})
    pd.DataFrame(deltas).to_csv(out / "ablation_deltas.csv", index=False)

    report = {"development_n": len(d), "relevant_n": int((d.gt_overall >= 2).sum()), "full": {"weights": FULL, "metrics": full_metrics}, "models": rows, "reoptimized_weights": reoptimized, "deltas_full_minus_reoptimized": deltas, "grid_combinations_per_ablation": 66}
    (out / "component_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

