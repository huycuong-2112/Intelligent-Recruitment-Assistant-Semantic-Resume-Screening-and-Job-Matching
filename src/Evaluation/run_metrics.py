from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

# 1. Add project root to sys.path to import modules from src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ground_truth import validate_ground_truth
from evaluator import evaluate

# 2. Define input and output paths
GT_JSON_PATH = project_root / "Data" / "GroundTruth" / "IT" / "jd_001.json"

GRID_CSV_PATH = project_root / "Data" / "Results" / "IT" / "Evaluation" / "development" / "jd_001_candidate_diagnostics_gridsearch.csv"
TF_CSV_PATH = project_root / "Data" / "Results" / "IT" / "Evaluation" / "development" / "predictions_tf.csv"
COS_CSV_PATH = project_root / "Data" / "Results" / "IT" / "Evaluation" / "development" / "predictions_cos.csv"

OUTPUT_DIR = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("RUNNING OFFICIAL METRIC RECOMPUTATION (DEVELOPMENT SPLIT)")
    print(f"Project Root: {project_root}")
    print("=" * 80)

    # 1. Read Ground Truth and filter for the 18 Development CVs
    with open(GT_JSON_PATH, "r", encoding="utf-8") as f:
        raw_gt_data = json.load(f)

    dev_candidates = [c for c in raw_gt_data["candidates"] if c.get("split") == "development"]
    gt_record = validate_ground_truth({
        "jd_id": raw_gt_data["jd_id"],
        "domain": raw_gt_data["domain"],
        "candidates": dev_candidates,
        "annotation_metadata": raw_gt_data.get("annotation_metadata", {}),
    })
    
    # Extract the exact 18 development CV IDs to act as a strict filter
    valid_cv_ids = {c["cv_id"] for c in dev_candidates}
    print(f"📂 Loaded Ground Truth: {len(gt_record.candidates)} candidates in development split.\n")

    # 2. Load the CSV files
    try:
        df_grid = pd.read_csv(GRID_CSV_PATH)
        df_tf = pd.read_csv(TF_CSV_PATH)
        df_cos = pd.read_csv(COS_CSV_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error loading file: {e}")
        print("Please ensure all 3 CSV files are placed in the correct directory.")
        sys.exit(1)

    # 3. Standardize input format AND strictly filter using valid_cv_ids
    # The 'if row["cv_id"] in valid_cv_ids' clause forces the 35-row CSVs down to the matching 18.
    
    sys_grid = [
        {"cv_id": row["cv_id"], "final_score": row["mdms_gridsearch"]} 
        for _, row in df_grid.iterrows() if row["cv_id"] in valid_cv_ids
    ]
    
    sys_tf = [
        {"cv_id": row["cv_id"], "final_score": row["score"]} 
        for _, row in df_tf.iterrows() if row["cv_id"] in valid_cv_ids
    ]
    
    sys_cos = [
        {"cv_id": row["cv_id"], "final_score": row["score"]} 
        for _, row in df_cos.iterrows() if row["cv_id"] in valid_cv_ids
    ]

    # 4. Run evaluate() from src/Evaluation/evaluator.py
    print("⚙️ Evaluating 3 models...")
    res_grid = evaluate(gt_record, sys_grid, method="mdms_gridsearch")
    res_tf = evaluate(gt_record, sys_tf, method="baseline_tfidf")
    res_cos = evaluate(gt_record, sys_cos, method="baseline_cos_sim")

    # 5. Aggregate results into a DataFrame
    metrics_rows = []
    for r in [res_grid, res_tf, res_cos]:
        m = r["metrics"]
        metrics_rows.append({
            "method": r["method"],
            "recall_5": round(m["recall@5"], 4),
            "recall_10": round(m["recall@10"], 4),
            "recall_15": round(m["recall@15"], 4),
            "ndcg_5": round(m["ndcg@5"], 4),
            "ndcg_10": round(m["ndcg@10"], 4),
            "ndcg_15": round(m["ndcg@15"], 4),
            "spearman": round(m["spearman"], 4),
            "mae_0_3": round(m["mae"], 4),
        })

    df_out = pd.DataFrame(metrics_rows)

    # 6. Print metrics table to terminal
    print("\n" + "=" * 80)
    print("RECOMPUTED METRICS RESULTS:")
    print("=" * 80)
    print(df_out.to_string(index=False))

    # 7. Save results to CSV
    metrics_csv_path = OUTPUT_DIR / "development_metrics.csv"
    df_out.to_csv(metrics_csv_path, index=False)
    print(f"\n💾 Saved results to: {metrics_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()