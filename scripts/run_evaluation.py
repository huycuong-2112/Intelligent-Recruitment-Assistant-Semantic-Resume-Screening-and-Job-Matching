from __future__ import annotations
import argparse, json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.Evaluation.ground_truth import load_ground_truth
from src.Evaluation.evaluator import evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument("--domain",required=True); p.add_argument("--jd-id",required=True); p.add_argument("--method",required=True); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); root=a.root
    gt=load_ground_truth(str(root/"Data/GroundTruth"/a.domain/f"{a.jd_id}.json")); result_dir=root/"Data/Results"/a.domain/("Matching" if a.method=="mdms" else f"Baselines/{a.method}")/a.jd_id; results=[json.loads(x.read_text(encoding="utf-8")) for x in result_dir.glob("*.json")]
    import yaml
    config=yaml.safe_load((root/"configs/mdms.yaml").read_text(encoding="utf-8")) or {}
    print(json.dumps(evaluate(gt,results,a.method,config),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
