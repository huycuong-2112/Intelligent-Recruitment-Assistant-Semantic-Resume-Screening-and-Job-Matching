from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# 1. Đường dẫn thư mục
project_root = Path(__file__).resolve().parent.parent.parent
EXCEL_PATH = project_root / "Data" / "Input" / "IT" / "groundtruth_overall_annotation_v2-1.xlsx"
OUTPUT_FILE = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001" / "annotation_agreement.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def calculate_agreement():
    print("=" * 80)
    print("MEASURING INTER-ANNOTATOR AGREEMENT (DEVELOPMENT SPLIT, N=18)")
    print("=" * 80)

    # 2. Đọc dữ liệu từ file Excel
    df_excel = pd.read_excel(EXCEL_PATH)
    dev_df = df_excel[df_excel["split"] == "development"].copy()

    a1 = dev_df["1_fit_label_0_to_3"].values
    a2 = dev_df["2_fit_label_0_to_3"].values
    a3 = dev_df["3_fit_label_0_to_3"].values
    n = len(dev_df)

    # 3. Tính toán các tỷ lệ đồng thuận cơ bản
    exact_3way = int(np.sum((a1 == a2) & (a2 == a3)))
    two_of_three = int(np.sum((a1 == a2) | (a2 == a3) | (a1 == a3)))
    three_way_disagree = int(np.sum((a1 != a2) & (a2 != a3) & (a1 != a3)))

    # 4. Tính toán Pairwise Kappa cho từng cặp
    pairs = [("A1", "A2", a1, a2), ("A1", "A3", a1, a3), ("A2", "A3", a2, a3)]
    pairwise_results = {}

    for n1, n2, y1, y2 in pairs:
        pairwise_results[f"{n1}_{n2}"] = {
            "exact_agreement": round(float(np.mean(y1 == y2)), 4),
            "unweighted_kappa": round(float(cohen_kappa_score(y1, y2)), 4),
            "linear_weighted_kappa": round(float(cohen_kappa_score(y1, y2, weights="linear")), 4),
            "quadratic_weighted_kappa": round(float(cohen_kappa_score(y1, y2, weights="quadratic")), 4),
        }

    # 5. Tính giá trị trung bình Kappa
    mean_linear = np.mean([v["linear_weighted_kappa"] for v in pairwise_results.values()])
    mean_quad = np.mean([v["quadratic_weighted_kappa"] for v in pairwise_results.values()])
    mean_unweighted = np.mean([v["unweighted_kappa"] for v in pairwise_results.values()])

    summary = {
        "n_samples": n,
        "exact_3way_count": exact_3way,
        "exact_3way_pct": round(float(exact_3way / n), 4),
        "two_of_three_count": two_of_three,
        "two_of_three_pct": round(float(two_of_three / n), 4),
        "three_way_disagree_count": three_way_disagree,
        "three_way_disagree_pct": round(float(three_way_disagree / n), 4),
        "pairwise": pairwise_results,
        "mean_linear_kappa": round(float(mean_linear), 4),
        "mean_quadratic_kappa": round(float(mean_quad), 4),
        "mean_unweighted_kappa": round(float(mean_unweighted), 4),
    }

    # 6. Lưu kết quả ra file JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"-> Đã xuất kết quả ra file: {OUTPUT_FILE}\n")
    return summary


if __name__ == "__main__":
    calculate_agreement()