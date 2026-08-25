"""Import immutable batched real inputs into canonical per-document artifacts and QA."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.Evaluation.ground_truth import validate_ground_truth

def write_records(records, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True); ids=[]; failures=[]; conflicts=[]
    for record in records:
        identifier=record.get("id") if isinstance(record,dict) else None
        if not identifier or identifier in ids: failures.append({"id":identifier,"reason":"missing or duplicate id"}); continue
        ids.append(identifier); path=out_dir/f"{identifier}.json"; payload=json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)
        if path.exists() and path.read_text(encoding="utf-8") != payload: conflicts.append(identifier)
        elif not path.exists(): path.write_text(payload,encoding="utf-8")
    return ids, failures, conflicts

def main():
    p=argparse.ArgumentParser(); p.add_argument("--domain",default="IT"); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); root=a.root; inp=root/"Data/Input"/a.domain
    cvs=json.loads(next(inp.glob("parsed_resumes_*.json")).read_text(encoding="utf-8")); jds=json.loads(next(inp.glob("parsed_jds_*.json")).read_text(encoding="utf-8")); cv_ids,cv_fail,cv_conf=write_records(cvs,root/"Data/Parsed"/a.domain/"CV"); jd_ids,jd_fail,jd_conf=write_records(jds,root/"Data/Parsed"/a.domain/"JD")
    xlsx=next(inp.glob("*.xlsx")); frame=pd.read_excel(xlsx); expected={"pair_id","jd_id","cv_id","split","overall"}; missing=expected-set(frame.columns)
    if missing: raise ValueError(f"missing GT columns: {sorted(missing)}")
    annotator_cols=[c for c in frame.columns if str(c).startswith(("1_","2_","3_"))]; pairs=set(); candidates=[]; invalid=[]
    for _,row in frame.iterrows():
        pair=str(row["pair_id"]); jd=str(row["jd_id"]); cv=str(row["cv_id"]); 
        if pair != f"{jd}__{cv}" or (jd,cv) in pairs or jd not in jd_ids or cv not in cv_ids: invalid.append({"pair_id":pair,"reason":"inconsistent, duplicate, or missing mapping"}); continue
        pairs.add((jd,cv)); item={"cv_id":cv,"split":str(row["split"]),"overall":int(row["overall"])}
        for index,column in enumerate(annotator_cols[:3],1): item[f"annotator_{index}"]=int(row[column])
        candidates.append(item)
    if invalid: raise ValueError(f"GT validation failures: {invalid[:3]}")
    by_jd={}
    for item,row in zip(candidates, frame.to_dict("records")): by_jd.setdefault(str(row["jd_id"]),[]).append(item)
    status_counts={}
    for cid in {x["cv_id"] for x in candidates}:
        record=json.loads((root/"Data/Parsed"/a.domain/"CV"/f"{cid}.json").read_text(encoding="utf-8")); status=str(record.get("source_status")); status_counts[status]=status_counts.get(status,0)+1
    qa={"parsed_cv_records":len(cv_ids),"parsed_jd_records":len(jd_ids),"gt_rows":len(frame),"unique_annotated_cvs":len({x["cv_id"] for x in candidates}),"missing_gt_cv_mappings":[x for x in frame.cv_id.astype(str) if x not in cv_ids],"missing_gt_jd_mappings":[x for x in frame.jd_id.astype(str) if x not in jd_ids],"duplicates":[],"grade_distribution":{str(k):int((frame.overall==k).sum()) for k in range(4)},"split_distribution":frame["split"].astype(str).value_counts().to_dict(),"annotated_cv_ids":sorted({x["cv_id"] for x in candidates}),"unannotated_cv_ids":sorted(set(cv_ids)-{x["cv_id"] for x in candidates}),"source_status_distribution":status_counts,"obvious_missing_core_fields":[],"warnings":cv_fail+jd_fail+cv_conf+jd_conf}
    qa_path=root/"Data/Results"/a.domain/"DataQA"/"jd_001_data_qa.json"; qa_path.parent.mkdir(parents=True,exist_ok=True); qa_path.write_text(json.dumps(qa,indent=2),encoding="utf-8")
    for jd,items in by_jd.items():
        artifact={"jd_id":jd,"domain":a.domain,"annotation_metadata":{"rubric_version":"v1","score_type":"ordinal_integer","score_min":0,"score_max":3,"final_label_field":"overall"},"candidates":items}; validate_ground_truth(artifact); path=root/"Data/GroundTruth"/a.domain/f"{jd}.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(artifact,indent=2),encoding="utf-8")
    manifest={"domain":a.domain,"jd_id":"jd_001","ground_truth_file":str(root/"Data/GroundTruth"/a.domain/"jd_001.json"),"candidate_ids":qa["annotated_cv_ids"],"n_candidates":len(qa["annotated_cv_ids"]),"gt_scale":"0-3 ordinal integer","relevant_threshold":2,"k_values":[5,10,15],"splits":qa["split_distribution"]}; (root/"Data/Results"/a.domain/"experiment_manifest_jd_001.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Imported CV={len(cv_ids)}, JD={len(jd_ids)}, GT rows={len(frame)}, annotated CV={len(candidates)}, unannotated CV={len(qa['unannotated_cv_ids'])}")
if __name__=="__main__": main()
