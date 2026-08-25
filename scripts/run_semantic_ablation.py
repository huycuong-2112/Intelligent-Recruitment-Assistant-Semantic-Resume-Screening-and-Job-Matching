from __future__ import annotations
import csv,json,statistics
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from src.Optimization.grid_search import _metrics

def main():
    root=Path(__file__).resolve().parents[1]; base=root/"Data/Results/IT"; d=pd.read_csv(base/"Evaluation/development/jd_001_candidate_diagnostics_continuous_experience.csv"); e=pd.read_csv(base/"Diagnostics/experience_jd_001_development_continuous.csv"); assert len(d)==18 and len(e)==18 and set(d.cv_id)==set(e.cv_id)
    d=d.merge(e[["cv_id","s_evidence_continuous"]],on="cv_id",validate="one_to_one"); out=base/"Ablation/Semantic/jd_001"; out.mkdir(parents=True,exist_ok=True)
    comps=["skill_score","experience_score","education_score","semantic_score"]; gt=d.gt_overall.tolist();
    corr=[]
    for a in comps+["s_evidence_continuous"]:
        row={"component":a}
        for b in comps+["s_evidence_continuous"]:
            row[f"spearman_{b}"]=spearmanr(d[a],d[b]).statistic; row[f"pearson_{b}"]=pearsonr(d[a],d[b]).statistic
        corr.append(row)
    pd.DataFrame(corr).to_csv(out/"component_correlations.csv",index=False)
    def metric(scores): return _metrics(scores,gt)
    full=[.4,.2,.1,.3]; equal=[.25]*4; frozen=[.4/.7,.2/.7,.1/.7,0]
    frozen_scores=[sum(frozen[i]*row[comps[i]] for i in range(3)) for _,row in d.iterrows()]; full_scores=[sum(full[i]*row[comps[i]] for i in range(4)) for _,row in d.iterrows()]
    grid=[]
    for a in range(11):
      for b in range(11-a):
       c = 10 - a - b
       weights=[a/10,b/10,c/10]; scores=[sum(weights[i]*row[comps[i]] for i in range(3)) for _,row in d.iterrows()]; m=metric(scores); grid.append({"w_skill":weights[0],"w_experience":weights[1],"w_education":weights[2],**m})
    def key(x): return (-float(x["ndcg@10"] or -1),-float(x["spearman"] or -1),float(x["ndcg@5"] or -1),float(x["mae_0_to_3"]),sum(abs(x[f"w_{n}"]-1/3) for n in ("skill","experience","education")),tuple(x[f"w_{n}"] for n in ("skill","experience","education")))
    grid=sorted(grid,key=key); best=grid[0]; pd.DataFrame(grid).to_csv(out/"no_semantic_grid_search.csv",index=False)
    metrics={"rule_based":json.load(open(base/"Evaluation/development/metrics_continuous_experience.json"))["rule_based"]["metrics"],"equal_mdms":metric([sum(.25*row[c] for c in comps) for _,row in d.iterrows()]),"full_tuned":metric(full_scores),"no_semantic_frozen":metric(frozen_scores),"no_semantic_reoptimized":{k:best[k] for k in ("recall@5","recall@10","recall@15","ndcg@5","ndcg@10","ndcg@15","spearman","mae_0_to_3")}}
    (out/"ablation_metrics.csv").write_text(pd.DataFrame([{"method":method,**vals} for method,vals in metrics.items()]).to_csv(index=False),encoding="utf-8")
    tuned_rank={cid:i+1 for i,(cid,_) in enumerate(sorted(zip(d.cv_id,full_scores),key=lambda x:x[1],reverse=True))}; no_scores=[sum(best[f"w_{n}"]*row[f"{n}_score"] for n in ("skill","experience","education")) for _,row in d.iterrows()]; no_rank={cid:i+1 for i,(cid,_) in enumerate(sorted(zip(d.cv_id,no_scores),key=lambda x:x[1],reverse=True))}; rank_rows=[]
    for (_,row),fs,ns in zip(d.iterrows(),full_scores,no_scores): rank_rows.append({"cv_id":row.cv_id,"gt_overall":row.gt_overall,"skill":row.skill_score,"experience":row.experience_score,"education":row.education_score,"semantic":row.semantic_score,"full_rank":tuned_rank[row.cv_id],"no_semantic_rank":no_rank[row.cv_id],"rank_change":no_rank[row.cv_id]-tuned_rank[row.cv_id]})
    pd.DataFrame(rank_rows).sort_values("rank_change",key=lambda x:x.abs(),ascending=False).to_csv(out/"rank_changes.csv",index=False)
    grid_df=pd.read_csv(base/"GridSearch/jd_001/grid_search_results.csv"); best_n=float(grid_df.iloc[0]["ndcg@10"]); near01=grid_df[grid_df["ndcg@10"]>=best_n-.01]; near02=grid_df[grid_df["ndcg@10"]>=best_n-.02]; stability={"winner":grid_df.iloc[0].to_dict(),"within_0.01":len(near01),"within_0.02":len(near02),"semantic_distribution_within_0.01":near01.w_semantic.value_counts().sort_index().to_dict(),"semantic_distribution_within_0.02":near02.w_semantic.value_counts().sort_index().to_dict()}; (out/"semantic_weight_stability.csv").write_text(pd.DataFrame({"region":["within_0.01","within_0.02"],"count":[len(near01),len(near02)],"semantic_min":[near01.w_semantic.min(),near02.w_semantic.min()],"semantic_max":[near01.w_semantic.max(),near02.w_semantic.max()],"semantic_mean":[near01.w_semantic.mean(),near02.w_semantic.mean()],"semantic_median":[near01.w_semantic.median(),near02.w_semantic.median()]}).to_csv(index=False),encoding="utf-8")
    report={"metrics":metrics,"correlations":corr,"stability":stability,"no_semantic_best":best,"full_minus_no_semantic":{k:metrics["full_tuned"][k]-metrics["no_semantic_reoptimized"][k] for k in metrics["full_tuned"]}}
    (out/"semantic_redundancy_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
