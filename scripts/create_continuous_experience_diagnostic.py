from __future__ import annotations
import csv, json, statistics
from pathlib import Path
def main():
    root=Path(__file__).resolve().parents[1]; base=root/"Data/Results/IT"; gt={r["cv_id"]:r["gt_overall"] for r in csv.DictReader((base/"Evaluation/development/jd_001_candidate_diagnostics_continuous_experience.csv").open(encoding="utf-8"))}; details={}
    for row in csv.DictReader((base/"Diagnostics/experience_responsibility_details_postfix.csv").open(encoding="utf-8")): details.setdefault(row["cv_id"],[]).append(float(row["raw_cosine"]))
    rows=[]
    for cid in gt:
        m=json.loads((base/"Matching/jd_001"/f"{cid}.json").read_text(encoding="utf-8")); values=details[cid]; e=m["experience"]; rows.append({"cv_id":cid,"gt_overall":gt[cid],"s_years":e["years"]["score"],"s_evidence_continuous":e["evidence"]["score"],"experience_score":e["score"],"experience_coverage":e["coverage"],"mean_best_similarity":statistics.mean(values),"max_best_similarity":max(values),"min_best_similarity":min(values),"median_best_similarity":statistics.median(values),"experience_status":e["evidence"]["status"]})
    out=base/"Diagnostics/experience_jd_001_development_continuous.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
if __name__=="__main__": main()
