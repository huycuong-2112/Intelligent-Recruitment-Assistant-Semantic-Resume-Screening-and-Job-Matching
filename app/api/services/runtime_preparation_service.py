"""Prepare confirmed runtime documents for downstream NLP matching inputs.

This service is deliberately orchestration-only: it invokes the locked
normalization, feature, chunk, profile and local embedding APIs.  It never
imports or executes matching, scoring, MDMS, XAI, or frontend freshness code.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

from app.api.core.config import settings
from src.Normalization.cv_normalizer import normalize_cv
from src.Normalization.jd_normalizer import normalize_jd
from src.Representation.feature_builder import build_cv_features, build_jd_features
from src.Representation.chunk_builder import build_cv_chunks, build_jd_chunks
from src.Representation.embedding_service import (
    EmbeddingService,
    build_cv_profile_text,
    build_jd_profile_text,
)

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class RuntimePreparationError(Exception):
    """Raised when a confirmed runtime document cannot be prepared."""


def _base(run_id: str, kind: str) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise RuntimePreparationError("invalid run_id")
    if kind not in {"cv", "jd"}:
        raise RuntimePreparationError("invalid document kind")
    root = Path(settings.RUNTIME_DATA_DIR).parent
    return root / ("resumes" if kind == "cv" else "jobs") / run_id / ("resumes" if kind == "cv" else "jobs")


def _load_confirmed(run_id: str, document_id: str, kind: str) -> tuple[dict[str, Any], Path, Path]:
    base = _base(run_id, kind)
    runtime_path = base / ("runtime_parsed_cv.json" if kind == "cv" else "runtime_parsed_jd.json")
    override_path = base / "confirm_override.json"
    if not runtime_path.is_file():
        raise RuntimePreparationError("confirmed runtime artifact not found; document must be confirmed first")
    if not override_path.is_file():
        raise RuntimePreparationError("confirmation artifact not found; document must be confirmed first")
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        override = json.loads(override_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimePreparationError("confirmation artifacts are malformed") from exc
    if not isinstance(runtime, dict) or runtime.get("id") != document_id:
        raise RuntimePreparationError("document_id does not match runtime artifact")
    if not isinstance(override, dict) or override.get("document_id") not in {None, document_id}:
        raise RuntimePreparationError("confirmation artifact identity mismatch")
    return runtime, runtime_path, override_path


def _jsonable(value: Any) -> Any:
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def _prepare(kind: str, run_id: str, document_id: str, domain: str, embedding_service: Any = None) -> dict[str, Any]:
    if not isinstance(domain, str) or not domain.strip():
        raise RuntimePreparationError("explicit domain is required")
    runtime, runtime_path, override_path = _load_confirmed(run_id, document_id, kind)
    # Runtime parser metadata may contain a generic domain.  The public domain
    # argument is authoritative without changing normalization semantics.
    source = dict(runtime)
    source.pop("domain", None)
    normalized = normalize_cv(source, domain=domain) if kind == "cv" else normalize_jd(source, domain=domain)
    if normalized.get("id") != document_id or normalized.get("domain") != domain:
        raise RuntimePreparationError("normalized runtime identity/domain mismatch")
    features_obj = build_cv_features(normalized) if kind == "cv" else build_jd_features(normalized)
    chunks_obj = build_cv_chunks(features_obj) if kind == "cv" else build_jd_chunks(features_obj)
    profile = build_cv_profile_text(features_obj) if kind == "cv" else build_jd_profile_text(features_obj)
    service = embedding_service or EmbeddingService()
    if kind == "cv":
        skills = list(features_obj.skills)
        skill_vectors = service.embed_batch(skills) if skills else []
        chunk_vectors = service.embed_batch([c.text for c in chunks_obj]) if chunks_obj else []
        artifact = {
            "id": document_id, "domain": domain,
            "model": {"name": service.model_name, "dimension": service.dimension},
            "skills": [{"skill": s, "source": [k for k, vals in features_obj.skill_provenance.items() if k != "all" and s in vals], "vector": v} for s, v in zip(skills, skill_vectors)],
            "experience_chunks": [{**dataclasses.asdict(c), "vector": v} for c, v in zip(chunks_obj, chunk_vectors)],
            "profile": {"text": profile, "vector": service.embed_text(profile) if profile else None},
        }
        counts = {"skills": len(skills), "experience_chunks": len(chunks_obj), "profile_chars": len(profile)}
        names = ("normalized_cv.json", "features_cv.json", "chunks_cv.json", "embeddings_cv.pkl")
    else:
        req, pref = list(features_obj.required_skills), list(features_obj.preferred_skills)
        req_vectors = service.embed_batch(req) if req else []
        pref_vectors = service.embed_batch(pref) if pref else []
        chunk_vectors = service.embed_batch([c.text for c in chunks_obj]) if chunks_obj else []
        artifact = {
            "id": document_id, "domain": domain,
            "model": {"name": service.model_name, "dimension": service.dimension},
            "required_skills": [{"skill": s, "vector": v} for s, v in zip(req, req_vectors)],
            "preferred_skills": [{"skill": s, "vector": v} for s, v in zip(pref, pref_vectors)],
            "responsibility_chunks": [{**dataclasses.asdict(c), "vector": v} for c, v in zip(chunks_obj, chunk_vectors)],
            "profile": {"text": profile, "vector": service.embed_text(profile) if profile else None},
        }
        counts = {"required_skills": len(req), "preferred_skills": len(pref), "responsibility_chunks": len(chunks_obj), "profile_chars": len(profile)}
        names = ("normalized_jd.json", "features_jd.json", "chunks_jd.json", "embeddings_jd.pkl")
    raw = json.dumps(runtime, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {"schema_version": "runtime_preparation_v1", "document_id": document_id, "run_id": run_id, "domain": domain,
                "source": {"runtime_parsed": str(runtime_path), "confirm_override": str(override_path)},
                "runtime_input_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "embedding": {"model_name": service.model_name, "dimension": service.dimension, "normalize_embeddings": getattr(service, "normalize_embeddings", True)}, "counts": counts}
    prepared = runtime_path.parent / "prepared"; prepared.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename, payload in zip(names[:3], (normalized, _jsonable(features_obj), [dataclasses.asdict(c) for c in chunks_obj])):
        path = prepared / filename; path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); paths[filename] = str(path)
    emb_path = prepared / names[3]
    with emb_path.open("wb") as fh: pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    paths[names[3]] = str(emb_path)
    manifest_path = prepared / "preparation_manifest.json"; manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"); paths["preparation_manifest.json"] = str(manifest_path)
    return {"run_id": run_id, "document_id": document_id, "domain": domain, "normalized": normalized, "features": _jsonable(features_obj), "chunks": [dataclasses.asdict(c) for c in chunks_obj], "embedding_artifact": artifact, "manifest": manifest, "artifact_paths": paths}


def prepare_cv_runtime(run_id: str, document_id: str, domain: str, embedding_service: Any = None) -> dict[str, Any]:
    return _prepare("cv", run_id, document_id, domain, embedding_service)


def prepare_jd_runtime(run_id: str, document_id: str, domain: str, embedding_service: Any = None) -> dict[str, Any]:
    return _prepare("jd", run_id, document_id, domain, embedding_service)
