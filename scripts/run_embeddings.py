"""Create inspectable embedding artifacts from normalized CV/JD JSON."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from src.Representation.chunk_builder import build_cv_chunks, build_jd_chunks
from src.Representation.embedding_service import EmbeddingService, build_cv_profile_text, build_jd_profile_text
from src.Representation.feature_builder import build_cv_features, build_jd_features


def _config(root: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load((root / "configs" / "mdms.yaml").read_text(encoding="utf-8")) or {}
        return data.get("embedding", {})
    except (OSError, yaml.YAMLError):
        return {}


def process(kind: str, domain: str, root: Path, service: EmbeddingService) -> tuple[int, int, list[str]]:
    source = root / "Data" / "Normalized" / domain / kind
    target = root / "Data" / "Embeddings" / domain / kind
    target.mkdir(parents=True, exist_ok=True)
    ok = failed = 0; paths: list[str] = []
    for path in sorted(source.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                if not isinstance(record, dict): raise ValueError("normalized record must be an object")
                if kind == "CV":
                    features = build_cv_features(record); chunks = build_cv_chunks(features); profile = build_cv_profile_text(features)
                    skill_values = features.skill_provenance.get("all", [])
                    skill_sources = {skill: [source for source, values in features.skill_provenance.items() if source != "all" and skill in values] for skill in skill_values}
                    skill_vectors = service.embed_batch(skill_values) if skill_values else []
                    chunk_vectors = service.embed_batch([chunk.text for chunk in chunks]) if chunks else []
                    artifact = {"id": features.id, "domain": features.domain, "model": {"name": service.model_name, "dimension": service.dimension}, "skills": [{"skill": skill, "source": skill_sources[skill], "vector": vector} for skill, vector in zip(skill_values, skill_vectors)], "experience_chunks": [{**chunk.__dict__, "vector": vector} for chunk, vector in zip(chunks, chunk_vectors)], "profile": {"text": profile, "vector": service.embed_text(profile) if profile else None}}
                else:
                    features = build_jd_features(record); chunks = build_jd_chunks(features); profile = build_jd_profile_text(features)
                    req_vectors = service.embed_batch(features.required_skills) if features.required_skills else []
                    pref_vectors = service.embed_batch(features.preferred_skills) if features.preferred_skills else []
                    chunk_vectors = service.embed_batch([chunk.text for chunk in chunks]) if chunks else []
                    artifact = {"id": features.id, "domain": features.domain, "model": {"name": service.model_name, "dimension": service.dimension}, "required_skills": [{"skill": skill, "vector": vector} for skill, vector in zip(features.required_skills, req_vectors)], "preferred_skills": [{"skill": skill, "vector": vector} for skill, vector in zip(features.preferred_skills, pref_vectors)], "responsibility_chunks": [{**chunk.__dict__, "vector": vector} for chunk, vector in zip(chunks, chunk_vectors)], "profile": {"text": profile, "vector": service.embed_text(profile) if profile else None}}
                out = target / f"{artifact['id'] or path.stem}.pkl"
                with out.open("wb") as handle: pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"{kind} {artifact['id']} | skills: {len(artifact.get('skills', artifact.get('required_skills', [])))} | evidence chunks: {len(chunks)} | profile chars: {len(profile)} | embedding dimension: {service.dimension} | saved: {out}")
                paths.append(str(out)); ok += 1
        except Exception as exc:
            print(f"[ERROR] {path}: {type(exc).__name__}: {exc}"); failed += 1
    return ok, failed, paths


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--domain", required=True); parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(); settings = _config(args.root)
    if "model_name" not in settings:
        raise SystemExit("Embedding model_name is missing from configs/mdms.yaml")
    service = EmbeddingService(settings["model_name"], settings.get("normalize_embeddings", True), settings.get("batch_size", 32))
    total = failed = 0; paths: list[str] = []
    for kind in ("CV", "JD"):
        count, errors, outputs = process(kind, args.domain, args.root, service); total += count; failed += errors; paths.extend(outputs)
    print(f"Summary: processed={total}, failed={failed}, outputs={len(paths)}")
    return 1 if failed else 0


if __name__ == "__main__": raise SystemExit(main())
