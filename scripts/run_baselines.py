from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from src.Baselines.rule_based_baseline import run_rule_based

def main():
    p=argparse.ArgumentParser(); p.add_argument("--domain",required=True); p.add_argument("--method",choices=["rule_based"],required=True); p.add_argument("--jd-id"); p.add_argument("--cv-ids",nargs="*"); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); root=a.root
    import yaml
    config=yaml.safe_load((root/"configs/mdms.yaml").read_text(encoding="utf-8")) or {}
    cvs={x.stem:json.loads(x.read_text(encoding="utf-8")) for x in (root/"Data/Normalized"/a.domain/"CV").glob("*.json")}; jds={x.stem:json.loads(x.read_text(encoding="utf-8")) for x in (root/"Data/Normalized"/a.domain/"JD").glob("*.json")}; out=root/"Data/Results"/a.domain/"Baselines"/a.method; count=0
    if a.jd_id: jds={key:value for key,value in jds.items() if key==a.jd_id}
    if a.cv_ids: cvs={key:value for key,value in cvs.items() if key in set(a.cv_ids)}
    for jid,jd in jds.items():
        for cid,cv in cvs.items():
            result=run_rule_based(jd,cv,config); path=out/jid/f"{cid}.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8"); count+=1
    print(f"Summary: processed={count}, output={out}"); return 0
if __name__=="__main__": raise SystemExit(main())
