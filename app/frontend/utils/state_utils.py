"""Pure helpers for confirmation freshness; no NLP or backend imports."""
from __future__ import annotations
import hashlib, json
from typing import Any

def feature_selection_fingerprint(features: list[dict] | None) -> str:
    items=[]
    for feature in features or []:
        if not isinstance(feature, dict): continue
        items.append({k: feature.get(k) for k in ("id","name","category","source_type","source_path","source_paths") if feature.get(k) is not None})
    items.sort(key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    payload=json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",",":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def confirmation_freshness(state: dict[str, Any], current_features: list[dict] | None) -> str:
    confirmed = state.get("confirmed_feature_fingerprint")
    if not state.get("runtime_parsed") or not state.get("confirm_status") or not confirmed:
        return "UNCONFIRMED"
    return "CONFIRMED" if feature_selection_fingerprint(current_features) == confirmed else "DIRTY"

def is_runtime_ready(state: dict[str, Any], current_features: list[dict] | None) -> bool:
    return bool(state.get("run_id") and state.get("document_id") and state.get("runtime_parsed") and state.get("confirm_status") in {"APPLIED","PARTIAL"} and state.get("confirmed_feature_fingerprint") and feature_selection_fingerprint(current_features) == state.get("confirmed_feature_fingerprint"))
