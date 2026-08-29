"""Fail-fast loader for the frozen runtime MDMS configuration."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

_KEYS = {"skill", "experience", "education", "semantic"}


class RuntimeMatchingConfigError(ValueError):
    pass


def load_runtime_matching_config(domain: str, root: Path | None = None) -> dict[str, Any]:
    if not isinstance(domain, str) or not domain.strip():
        raise RuntimeMatchingConfigError("explicit domain is required")
    config_path = (root or Path(__file__).resolve().parents[3]) / "configs" / "mdms.yaml"
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeMatchingConfigError(f"cannot load runtime config: {exc}") from exc
    configured_domain = data.get("domain")
    if configured_domain != domain:
        raise RuntimeMatchingConfigError(f"runtime config domain {configured_domain!r} is incompatible with requested domain {domain!r}")
    mdms = data.get("mdms") or {}
    weights = mdms.get("runtime_weights")
    if not isinstance(weights, dict) or set(weights) != _KEYS:
        raise RuntimeMatchingConfigError("mdms.runtime_weights must contain exactly skill, experience, education, semantic")
    validated: dict[str, float] = {}
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise RuntimeMatchingConfigError(f"invalid runtime weight for {key}")
        validated[key] = float(value)
    if abs(sum(validated.values()) - 1.0) > 1e-6:
        raise RuntimeMatchingConfigError("runtime weights must sum to 1")
    metadata = mdms.get("runtime_weights_metadata")
    if not isinstance(metadata, dict):
        raise RuntimeMatchingConfigError("runtime_weights_metadata is required")
    return {"weights": validated, "version": metadata.get("version"), "selection_scope": metadata.get("selected_on", "development"), "source": metadata.get("source"), "source_jd": metadata.get("source_jd"), "blind_evaluated": metadata.get("blind_evaluated")}


def load_runtime_mdms_weights(domain: str, root: Path | None = None) -> dict[str, float]:
    return load_runtime_matching_config(domain, root)["weights"]
