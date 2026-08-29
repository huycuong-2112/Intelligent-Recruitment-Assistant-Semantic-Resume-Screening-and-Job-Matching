"""Deterministic application-layer Confirm → runtime document overrides."""
from __future__ import annotations
from copy import deepcopy
import hashlib, re
from typing import Any
from src.Normalization.skill_normalizer import normalize_skill

_PATH = re.compile(r"^parsed_data(?:\.([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?)+$")
_SCALARS = {"education_degree", "education_field", "experience_years", "required_degree", "min_experience_years"}
_CV_MANUAL = {"Skills": "skills"}
_JD_MANUAL = {"Required Skills": "required_skills", "Preferred Skills": "preferred_skills", "Responsibilities": "responsibilities", "Certifications": "required_certifications"}
_VALID_MANUAL_TYPES = {"Skills": {"skill"}, "Education": {"degree", "field"}, "Experience": {"role", "responsibility"}, "Projects": {"project_name", "project_evidence", "project"}}
_VALID_JD_TYPES = {"Required Skills": {"required_skill"}, "Preferred Skills": {"preferred_skill"}, "Responsibilities": {"responsibility"}, "Certifications": {"certification"}, "Education": {"required_degree", "preferred_field"}}

def _doc_id(parsed: dict[str, Any]) -> str:
    return str(parsed.get("id", ""))

def _manual_id(document_id: str, feature: dict[str, Any], index: int) -> str:
    raw = f"{document_id}|{feature.get('category','')}|{str(feature.get('name','')).strip().casefold()}|{index}"
    return f"{document_id}_manual_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"

def build_confirm_override(document_id: str, original_features: list[dict], confirmed_features: list[dict]) -> dict[str, Any]:
    if not document_id: raise ValueError("document_id is required")
    if not isinstance(original_features, list) or not isinstance(confirmed_features, list): raise ValueError("features must be lists")
    original = {}; confirmed = {}; added = []
    for f in original_features:
        if not isinstance(f, dict) or not f.get("id"): raise ValueError("malformed original feature")
        if not str(f["id"]).startswith(document_id + "_"): raise ValueError("feature belongs to another document")
        if f["id"] in original: raise ValueError("duplicate feature id")
        original[f["id"]] = f
    semantic_seen = {(str(f.get("category")), _semantic_key(f)) for f in original_features}
    for i, f in enumerate(confirmed_features):
        if not isinstance(f, dict) or not f.get("name") or not f.get("category"): raise ValueError("malformed confirmed feature")
        if f.get("source_type") == "manual_ui":
            category = str(f.get("category")); feature_type = f.get("feature_type") or ("skill" if category == "Skills" else None)
            # Legacy untyped JD/manual Education entries remain compatible;
            # typed CV entries are strictly category-safe.
            if category in _VALID_MANUAL_TYPES and feature_type is not None and feature_type not in _VALID_MANUAL_TYPES[category]:
                raise ValueError("invalid manual feature category/type")
            if category in _VALID_JD_TYPES and feature_type in {"required_skill", "preferred_skill", "responsibility", "certification", "required_degree", "preferred_field"} and feature_type not in _VALID_JD_TYPES[category]:
                raise ValueError("invalid manual feature category/type")
            f = {**f, "feature_type": feature_type}
            key = (str(f.get("category")), _semantic_key(f))
            if key in semantic_seen: raise ValueError(f"{f.get('name')} already exists in {f.get('category')}.")
            semantic_seen.add(key)
            item = deepcopy(f); item["id"] = item.get("id") or _manual_id(document_id, item, i); added.append(item); continue
        fid = f.get("id")
        if not fid or fid not in original: raise ValueError("confirmed feature id is not present in original features")
        if fid in confirmed: raise ValueError("duplicate confirmed feature id")
        confirmed[fid] = f
    removed = [fid for fid in original if fid not in confirmed]
    return {"document_id": document_id, "kept_feature_ids": list(confirmed), "removed_feature_ids": removed,
            "removed_features": [deepcopy(original[fid]) for fid in removed], "added_features": added}

def _semantic_key(feature: dict[str, Any]) -> str:
    value = str(feature.get("name", "")).strip()
    if feature.get("category") == "Skills": return normalize_skill(value).casefold()
    if feature.get("category") == "Education":
        value = re.sub(r"^(degree|field(?: of study)?):\s*", "", value, flags=re.I)
    return " ".join(value.casefold().split())

def _resolve(data: dict[str, Any], path: str):
    if not isinstance(path, str) or not path.startswith("parsed_data.") or "__" in path or ".." in path or not _PATH.match(path): raise ValueError(f"unsafe or unsupported source path: {path}")
    tokens = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", path[len("parsed_data"):])
    cur: Any = data.get("parsed_data")
    if not isinstance(cur, dict): raise ValueError("parsed_data is missing")
    for key, idx in tokens:
        if not isinstance(cur, dict) or key not in cur: raise ValueError(f"source path not found: {path}")
        cur = cur[key]
        if idx:
            if not isinstance(cur, list) or int(idx) >= len(cur): raise ValueError(f"source path not found: {path}")
            cur = cur[int(idx)]
    return tokens

def _remove(data: dict[str, Any], path: str):
    tokens = _resolve(data, path); cur: Any = data["parsed_data"]
    for key, idx in tokens[:-1]:
        cur = cur[key]; cur = cur[int(idx)] if idx else cur
    key, idx = tokens[-1]
    if idx:
        del cur[key][int(idx)]
    elif key in _SCALARS or not isinstance(cur.get(key), list):
        cur[key] = None
    else:
        raise ValueError(f"cannot remove unindexed list field: {path}")

def _apply(parsed: dict[str, Any], override: dict[str, Any], manual_map: dict[str, str], kind: str) -> dict[str, Any]:
    if not isinstance(override, dict) or not override.get("document_id"): raise ValueError("invalid override: document_id is required")
    if _doc_id(parsed) != override["document_id"]: raise ValueError("override document_id does not match parsed document")
    runtime = deepcopy(parsed); applied=[]; unsupported=[]
    # Removed features are supplied by the caller through the original feature metadata.
    for feature in override.get("removed_features", []):
        paths = feature.get("source_paths") or ([feature.get("source_path")] if feature.get("source_path") else [])
        # Remove list entries from highest index first to avoid index shifts.
        for path in sorted(paths, key=lambda p: [int(x) for x in re.findall(r"\[(\d+)\]", p or "")], reverse=True): _remove(runtime, path)
        applied.append({"feature_id": feature.get("id"), "action": "removed", "source_paths": paths})
    for feature in override.get("added_features", []):
        category = feature.get("category")
        target = manual_map.get(category)
        if kind == "cv" and category == "Education":
            field_type = feature.get("feature_type")
            target = "education_degree" if field_type == "degree" else "education_field" if field_type == "field" else None
            if target:
                runtime.setdefault("parsed_data", {})[target] = str(feature["name"]).strip()
                applied.append({"feature_id": feature.get("id"), "action": "added", "category": category, "target_path": f"parsed_data.{target}", "name": feature["name"]})
                continue
        if kind == "cv" and category == "Experience":
            field_type = feature.get("feature_type")
            if field_type in {"role", "responsibility"}:
                work = runtime.setdefault("parsed_data", {}).setdefault("work_experience", [])
                if field_type == "role": work.append({"role": str(feature["name"]).strip(), "responsibilities_and_impact": []})
                else:
                    if not work: work.append({"role": None, "responsibilities_and_impact": []})
                    work[-1].setdefault("responsibilities_and_impact", []).append(str(feature["name"]).strip())
                applied.append({"feature_id": feature.get("id"), "action": "added", "category": category, "name": feature["name"]})
                continue
        if kind == "cv" and category == "Projects":
            if feature.get("feature_type") in {"project", "project_name", "project_evidence"}:
                runtime.setdefault("parsed_data", {}).setdefault("projects", []).append({"name": str(feature["name"]).strip(), "description": str(feature["name"]).strip(), "technologies": []})
                applied.append({"feature_id": feature.get("id"), "action": "added", "category": category, "name": feature["name"]})
                continue
        if kind == "jd" and category == "Education" and feature.get("feature_type") in {"required_degree", "preferred_field"}:
            target = "required_degree" if feature["feature_type"] == "required_degree" else "preferred_fields"
            values = runtime.setdefault("parsed_data", {}).setdefault(target, []) if target.endswith("fields") else runtime.setdefault("parsed_data", {})
            if target.endswith("fields"): values.append(str(feature["name"]).strip())
            else: values[target] = str(feature["name"]).strip()
            applied.append({"feature_id": feature.get("id"), "action": "added", "category": category, "target_path": f"parsed_data.{target}", "name": feature["name"]})
            continue
        if not target:
            unsupported.append({"feature_id": feature.get("id"), "action": "unsupported", "category": feature.get("category"), "reason": "no safe structured target"}); continue
        values = runtime.setdefault("parsed_data", {}).setdefault(target, [])
        if not any(isinstance(v, str) and v.strip().casefold() == str(feature["name"]).strip().casefold() for v in values): values.append(str(feature["name"]).strip())
        applied.append({"feature_id": feature.get("id"), "action": "added", "category": feature.get("category"), "target_path": f"parsed_data.{target}", "name": feature["name"]})
    return {"runtime_document": runtime, "applied_actions": applied, "unsupported_actions": unsupported}

def apply_cv_override(parsed_cv: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]: return _apply(parsed_cv, override, _CV_MANUAL, "cv")
def apply_jd_override(parsed_jd: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]: return _apply(parsed_jd, override, _JD_MANUAL, "jd")
