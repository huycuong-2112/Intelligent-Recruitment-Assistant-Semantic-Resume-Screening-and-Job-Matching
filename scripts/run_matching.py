"""Run deterministic CV/JD matching over existing normalized and embedding artifacts."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from src.Matching.education_matcher import match_education
from src.Matching.experience_matcher import match_experience
from src.Matching.mdms import aggregate_mdms
from src.Matching.semantic_matcher import match_semantic
from src.Matching.skill_matcher import match_skills

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--domain", required=True); parser.add_argument("--jd-id"); parser.add_argument("--cv-ids", nargs="*"); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = parser.parse_args()
    root, domain = args.root, args.domain
    norm_cv, norm_jd = root / "Data/Normalized" / domain / "CV", root / "Data/Normalized" / domain / "JD"
    emb_cv, emb_jd = root / "Data/Embeddings" / domain / "CV", root / "Data/Embeddings" / domain / "JD"
    out_root = root / "Data/Results" / domain / "Matching"; out_root.mkdir(parents=True, exist_ok=True)
    config = {}
    try:
        import yaml; config = yaml.safe_load((root / "configs/mdms.yaml").read_text(encoding="utf-8")) or {}
    except Exception as exc: print(f"[ERROR] config: {exc}"); return 1
    cvs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in norm_cv.glob("*.json")}; jds = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in norm_jd.glob("*.json")}
    if args.jd_id: jds = {key: value for key, value in jds.items() if key == args.jd_id}
    if args.cv_ids: cvs = {key: value for key, value in cvs.items() if key in set(args.cv_ids)}
    cv_artifacts = {p.stem: pickle.loads(p.read_bytes()) for p in emb_cv.glob("*.pkl")}; jd_artifacts = {p.stem: pickle.loads(p.read_bytes()) for p in emb_jd.glob("*.pkl")}
    weights = config.get("mdms", {}).get("provisional_weights", {"skill": .25, "experience": .25, "education": .25, "semantic": .25})
    failed = 0
    for jd_id, jd in jds.items():
        for cv_id, cv in cvs.items():
            try:
                skill = match_skills(jd, cv, cv_artifacts.get(cv_id), config, jd_artifacts.get(jd_id)); experience = match_experience(jd, cv, cv_artifacts.get(cv_id), jd_artifacts.get(jd_id), config); education = match_education(jd, cv, config); semantic = match_semantic(cv_artifacts.get(cv_id), jd_artifacts.get(jd_id)); mdms = aggregate_mdms({"skill": skill, "experience": experience, "education": education, "semantic": semantic}, weights)
                result = {"jd_id": jd_id, "cv_id": cv_id, "score_0_1": mdms.get("final_score"), "status": mdms.get("status"), "skill": skill, "experience": experience, "education": education, "semantic": semantic, "mdms": {**mdms, "weights": weights}}
                path = out_root / jd_id / f"{cv_id}.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"JD: {jd_id} | CV: {cv_id} | Skill: {skill['score']} | Experience: {experience['score']} | Education: {education['score']} | Semantic: {semantic.get('score')} | MDMS: {mdms['final_score']}")
            except Exception as exc: print(f"[ERROR] {jd_id}/{cv_id}: {type(exc).__name__}: {exc}"); failed += 1
    print(f"Summary: processed={len(jds)*len(cvs)-failed}, failed={failed}, output={out_root}")
    return 1 if failed else 0
if __name__ == "__main__": raise SystemExit(main())
