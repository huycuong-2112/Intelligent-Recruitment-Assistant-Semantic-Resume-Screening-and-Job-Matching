from __future__ import annotations
import csv, json, pickle, statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from src.Matching.common import cosine

def main():
    root=Path(__file__).resolve().parents[1]; gt={x["cv_id"]:x["gt_overall"] for x in csv.DictReader((root/"Data/Results/IT/Evaluation/development/jd_001_candidate_diagnostics.csv").open(encoding="utf-8"))}; ids=list(gt); jd=pickle.loads((root/"Data/Embeddings/IT/JD/jd_001.pkl").read_bytes()); responsibilities=[item for item in jd["responsibility_chunks"] if item.get("source_type")=="responsibility"]; threshold=.6; rows=[]; detail=[]
    for cid in ids:
        normalized=json.loads((root/"Data/Normalized/IT/CV"/f"{cid}.json").read_text(encoding="utf-8")); artifact=pickle.loads((root/"Data/Embeddings/IT/CV"/f"{cid}.pkl").read_bytes()); official=json.loads((root/"Data/Results/IT/Matching/jd_001"/f"{cid}.json").read_text(encoding="utf-8")); chunks=artifact.get("experience_chunks",[]); sims=[]
        for responsibility in responsibilities:
            candidates=[(cosine(responsibility["vector"],chunk["vector"]),chunk) for chunk in chunks]
            best=max(candidates,key=lambda x:x[0]) if candidates else (None,None); sims.append(best[0])
            detail.append({"cv_id":cid,"responsibility":responsibility["text"],"best_evidence":best[1].get("text") if best[1] else None,"source":best[1].get("source_type") if best[1] else None,"raw_cosine":best[0],"threshold":threshold,"accepted":bool(best[0] is not None and best[0]>=threshold),"contribution":best[0] if best[0] is not None and best[0]>=threshold else 0.0})
        valid=[x for x in sims if x is not None]; evidence_reconstructed=sum(x for x in sims if x is not None and x>=threshold)/len(sims) if sims else None; rows.append({"cv_id":cid,"gt_overall":gt.get(cid),"s_years":official["experience"]["years"]["score"],"s_evidence_official":official["experience"]["evidence"]["score"],"experience_score_official":official["experience"]["score"],"coverage_official":official["experience"]["coverage"],"evidence_status_official":official["experience"]["evidence"]["status"],"max_raw_similarity":max(valid) if valid else None,"mean_best_similarity":statistics.mean(valid) if valid else None,"responsibilities_above_threshold":sum(x>=threshold for x in valid),"s_evidence_reconstructed":evidence_reconstructed})
    out=root/"Data/Results/IT/Diagnostics"; out.mkdir(parents=True,exist_ok=True)
    with (out/"experience_jd_001_development.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (out/"experience_responsibility_details.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=detail[0].keys()); w.writeheader(); w.writerows(detail)
    print(f"Wrote {len(rows)} candidate rows and {len(detail)} responsibility rows to {out}")
if __name__=="__main__": main()
