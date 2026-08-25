from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.Optimization.grid_search import search, _metrics

def main():
    root=Path(__file__).resolve().parents[1]; base=root/"Data/Results/IT"; source=base/"Evaluation/development/jd_001_candidate_diagnostics_continuous_experience.csv"; rows=list(csv.DictReader(source.open(encoding="utf-8")))
    required=["cv_id","gt_overall","skill_score","experience_score","education_score","semantic_score"]
    if any(any(row.get(key) in (None,"") for key in required) for row in rows): raise ValueError("missing grid-search component score")
    normalized=[{"cv_id":r["cv_id"],"gt_overall":int(r["gt_overall"]),"skill_score":float(r["skill_score"]),"experience_score":float(r["experience_score"]),"education_score":float(r["education_score"]),"semantic_score":float(r["semantic_score"])} for r in rows]
    ordered,best=search(normalized); out=base/"GridSearch/jd_001"; out.mkdir(parents=True,exist_ok=True)
    with (out/"grid_search_results.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=ordered[0].keys()); w.writeheader(); w.writerows(ordered)
    payload={"target_jd":"jd_001","development_candidate_count":len(rows),"objective":"maximize development ndcg@10","grid_step":0.1,"best_weights":{k:best[k] for k in ("w_skill","w_experience","w_education","w_semantic")},"development_metrics":{k:best[k] for k in ("recall@5","recall@10","recall@15","ndcg@5","ndcg@10","ndcg@15","spearman","mae_0_to_3")},"tie_break_logic":["higher ndcg@10","higher spearman","higher ndcg@5","lower mae_0_to_3","smaller L1 distance from equal weights","lexicographic weights"],"metric_tolerance":1e-9,"source_component_score_artifact":str(source),"selected_on":"development","blind_evaluated":False,"timestamp_utc":datetime.now(timezone.utc).isoformat()}; (out/"best_weights.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    (out/"frozen_weights.json").write_text(json.dumps({"target_jd":"jd_001","weights":payload["best_weights"],"selected_on":"development","blind_evaluated":False},indent=2),encoding="utf-8")
    equal=[.25]*4; best_scores=[sum(equal[i]*row[f"{('skill','experience','education','semantic')[i]}_score"] for i in range(4)) for row in normalized]; winner_weights=[best[f"w_{name}"] for name in ('skill','experience','education','semantic')]; winner_scores=[sum(winner_weights[i]*row[f"{('skill','experience','education','semantic')[i]}_score"] for i in range(4)) for row in normalized]
    def rank(values): return {row["cv_id"]:index+1 for index,(row,_) in enumerate(sorted(zip(normalized,values),key=lambda pair:pair[1],reverse=True))}
    eq_rank,win_rank=rank(best_scores),rank(winner_scores); diag=[]
    for row,eq,win in zip(normalized,best_scores,winner_scores): diag.append({**row,"mdms_equal_weight":eq,"mdms_gridsearch":win,"mdms_equal_rank":eq_rank[row["cv_id"]],"mdms_gridsearch_rank":win_rank[row["cv_id"]],"mdms_gridsearch_0_to_3":win*3,"absolute_error_equal":abs(row["gt_overall"]-eq*3),"absolute_error_gridsearch":abs(row["gt_overall"]-win*3)})
    with (base/"Evaluation/development/jd_001_candidate_diagnostics_gridsearch.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=diag[0].keys()); w.writeheader(); w.writerows(diag)
    near01=[x for x in ordered if best["ndcg@10"]-float(x["ndcg@10"])<=.01]; near02=[x for x in ordered if best["ndcg@10"]-float(x["ndcg@10"])<=.02]; all_positive=[x for x in ordered if all(x[f"w_{name}"]>0 for name in ('skill','experience','education','semantic'))][0]
    (out/"stability.json").write_text(json.dumps({"within_0.01":len(near01),"within_0.02":len(near02),"ranges_within_0.02":{name:[min(x[f"w_{name}"] for x in near02),max(x[f"w_{name}"] for x in near02)] for name in ('skill','experience','education','semantic')},"best_all_positive":all_positive},indent=2),encoding="utf-8")
    print(json.dumps({"combinations":len(ordered),"best":best},indent=2))
if __name__=="__main__": main()
