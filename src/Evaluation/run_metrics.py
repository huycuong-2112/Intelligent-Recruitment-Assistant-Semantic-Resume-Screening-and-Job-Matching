from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

# 1. Thêm đường dẫn project vào sys.path để import module trong src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ground_truth import validate_ground_truth
from evaluator import evaluate

# 2. Định nghĩa các đường dẫn file input và output
GT_JSON_PATH = project_root / "Data" / "GroundTruth" / "IT" / "jd_001.json"
CONT_CSV_PATH = project_root / "Data" / "Results" / "IT" / "Evaluation" / "development" / "jd_001_candidate_diagnostics_continuous_experience.csv"
GRID_CSV_PATH = project_root / "Data" / "Results" / "IT" / "Evaluation" / "development" / "jd_001_candidate_diagnostics_gridsearch.csv"
OUTPUT_DIR = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("RUNNING OFFICIAL METRIC RECOMPUTATION (DEVELOPMENT SPLIT)")
    print(f"Project Root: {project_root}")
    print("=" * 80)

    # 1. Đọc Ground Truth và lọc chỉ lấy 18 CV tập Development
    with open(GT_JSON_PATH, "r", encoding="utf-8") as f:
        raw_gt_data = json.load(f)

    dev_candidates = [c for c in raw_gt_data["candidates"] if c.get("split") == "development"]
    gt_record = validate_ground_truth({
        "jd_id": raw_gt_data["jd_id"],
        "domain": raw_gt_data["domain"],
        "candidates": dev_candidates,
        "annotation_metadata": raw_gt_data.get("annotation_metadata", {}),
    })
    print(f"📂 Loaded Ground Truth: {len(gt_record.candidates)} candidates in development split.\n")

    # 2. Đọc file CSV điểm của các model
    df_cont = pd.read_csv(CONT_CSV_PATH)
    df_grid = pd.read_csv(GRID_CSV_PATH)

    # Chuẩn hóa format đầu vào cho hàm evaluate()
    sys_rule = [{"cv_id": row["cv_id"], "final_score": row["rule_based_score"]} for _, row in df_cont.iterrows()]
    sys_equal = [{"cv_id": row["cv_id"], "final_score": row["mdms_equal_weight"]} for _, row in df_cont.iterrows()]
    sys_tuned = [{"cv_id": row["cv_id"], "final_score": row["mdms_gridsearch"]} for _, row in df_grid.iterrows()]

    # 3. Chạy hàm evaluate() từ src/Evaluation/evaluator.py
    print("⚙️ Evaluating 3 models...")
    res_rule = evaluate(gt_record, sys_rule, method="rule_based_v1")
    res_equal = evaluate(gt_record, sys_equal, method="mdms_equal_v1")
    res_tuned = evaluate(gt_record, sys_tuned, method="mdms_tuned_v1")

    # 4. Gom kết quả vào DataFrame
    metrics_rows = []
    for r in [res_rule, res_equal, res_tuned]:
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

    # 5. In bảng kết quả ra terminal
    print("\n" + "=" * 80)
    print("RECOMPUTED METRICS RESULTS:")
    print("=" * 80)
    print(df_out.to_string(index=False))

    # 6. Lưu file CSV
    metrics_csv_path = OUTPUT_DIR / "development_metrics.csv"
    df_out.to_csv(metrics_csv_path, index=False)
    print(f"\n💾 Saved results to: {metrics_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()