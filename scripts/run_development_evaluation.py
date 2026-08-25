"""Produce pre-tuning development-only metrics and diagnostics for jd_001."""
from __future__ import annotations
import csv, json, math, shutil
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.Evaluation.ground_truth import load_ground_truth
from src.Evaluation.evaluator import evaluate
from src.Evaluation.metrics import mean_absolute_error

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--suffix",default=""); args=parser.parse_args(); root=Path(__file__).resolve().parents[1]; base=root/"Data/Results/IT"; manifest=json.loads((base/"experiment_manifest_jd_001.json").read_text()); gt=load_ground_truth(str(root/"Data/GroundTruth/IT/jd_001.json")); dev=[x for x in gt.candidates if x.split=="development"]; ids={x.cv_id for x in dev}; results=[]
    for cid in ids:
        results.append(json.loads((base/"Matching/jd_001"/f"{cid}.json").read_text(encoding="utf-8")))
    baseline=[json.loads((base/"Baselines/rule_based/jd_001"/f"{cid}.json").read_text(encoding="utf-8")) for cid in ids]
    config=yaml.safe_load((root/"configs/mdms.yaml").read_text()) or {}; metrics_mdms=evaluate(type(gt)(gt.jd_id,gt.domain,tuple(dev),gt.annotation_metadata),results,"mdms_equal_weight",config); metrics_rule=evaluate(type(gt)(gt.jd_id,gt.domain,tuple(dev),gt.annotation_metadata),baseline,"rule_based",config)
    evaluation_dir=base/"Evaluation/development"; evaluation_dir.mkdir(parents=True,exist_ok=True); (evaluation_dir/f"metrics{args.suffix}.json").write_text(json.dumps({"rule_based":metrics_rule,"mdms_equal_weight":metrics_mdms},indent=2))
    gt_map={x.cv_id:x for x in dev}; rows=[]
    for item in results:
        cid=item["cv_id"]; b=next(x for x in baseline if x["cv_id"]==cid); mdms=item["score_0_1"]; row={"cv_id":cid,"gt_overall":gt_map[cid].overall,"skill_score":item["skill"]["score"],"skill_coverage":item["skill"]["coverage"],"experience_score":item["experience"]["score"],"experience_coverage":item["experience"]["coverage"],"education_score":item["education"]["score"],"education_coverage":item["education"]["coverage"],"semantic_score":item["semantic"]["score"],"semantic_coverage":item["semantic"]["coverage"],"mdms_equal_weight":mdms,"mdms_0_to_3":mdms*3,"rule_based_score":b["score_0_1"],"overall_coverage":item["mdms"].get("coverage"),"absolute_error_mdms":abs(gt_map[cid].overall-mdms*3),"absolute_error_rule":abs(gt_map[cid].overall-b["score_0_1"]*3)}; rows.append(row)
    for key in ("mdms_equal_weight","rule_based_score"): order=sorted(rows,key=lambda x:x[key],reverse=True); ranks={x["cv_id"]:i+1 for i,x in enumerate(order)}; [x.update({"mdms_rank" if key.startswith("mdms") else "rule_based_rank":ranks[x["cv_id"]]}) for x in rows]
    gt_order=sorted(rows,key=lambda x:x["gt_overall"],reverse=True); ranks={x["cv_id"]:i+1 for i,x in enumerate(gt_order)}; [x.update({"gt_rank":ranks[x["cv_id"]]}) for x in rows]
    with (evaluation_dir/f"jd_001_candidate_diagnostics{args.suffix}.csv").open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    errors={"largest_mdms":sorted(rows,key=lambda x:x["absolute_error_mdms"],reverse=True)[:5],"largest_rule_based":sorted(rows,key=lambda x:x["absolute_error_rule"],reverse=True)[:5],"low_coverage":[x for x in rows if x["overall_coverage"] is not None and x["overall_coverage"]<1]}; (evaluation_dir/"error_analysis.json").write_text(json.dumps(errors,indent=2))
    agreement={"three_way_exact":sum(x.annotator_1==x.annotator_2==x.annotator_3 for x in dev),"two_of_three":sum(len({x.annotator_1,x.annotator_2,x.annotator_3})==2 for x in dev),"three_different":sum(len({x.annotator_1,x.annotator_2,x.annotator_3})==3 for x in dev)}
    try:
        from sklearn.metrics import cohen_kappa_score
        agreement["weighted_kappa"]={"a1_a2":cohen_kappa_score([x.annotator_1 for x in dev],[x.annotator_2 for x in dev],weights="quadratic"),"a1_a3":cohen_kappa_score([x.annotator_1 for x in dev],[x.annotator_3 for x in dev],weights="quadratic"),"a2_a3":cohen_kappa_score([x.annotator_2 for x in dev],[x.annotator_3 for x in dev],weights="quadratic")}
    except Exception: pass
    (evaluation_dir/"annotation_agreement.json").write_text(json.dumps(agreement,indent=2)); snapshot=base/"PreTuning/jd_001"; snapshot.mkdir(parents=True,exist_ok=True); shutil.copy2(root/"configs/mdms.yaml",snapshot/"mdms.yaml"); shutil.copy2(evaluation_dir/"metrics.json",snapshot/"metrics.json"); shutil.copy2(evaluation_dir/"jd_001_candidate_diagnostics.csv",snapshot/"candidate_diagnostics.csv"); (snapshot/"run_metadata.json").write_text(json.dumps({"timestamp_utc":datetime.now(timezone.utc).isoformat(),"candidate_ids":sorted(ids),"development_ids":sorted(ids),"blind_ids":sorted({x.cv_id for x in gt.candidates if x.split=="blind_test"}),"weights":config["mdms"]["provisional_weights"]},indent=2))
    print(json.dumps({"development_n":len(dev),"metrics": {"rule_based":metrics_rule["metrics"],"mdms_equal_weight":metrics_mdms["metrics"]},"agreement":agreement},indent=2))
if __name__=="__main__": main()
