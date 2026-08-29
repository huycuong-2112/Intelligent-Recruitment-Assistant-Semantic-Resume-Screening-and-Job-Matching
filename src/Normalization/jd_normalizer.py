"""Normalize parsed job-description records deterministically."""

from __future__ import annotations

from typing import Any
import re

from .skill_normalizer import normalize_skills
from .experience import parse_experience


_GENERIC_FIELD_QUALIFIERS = {"related field", "related fields", "relevant field", "relevant fields", "related discipline", "related disciplines", "relevant discipline", "relevant disciplines", "equivalent field", "equivalent fields"}

def _atomic_fields(values: Any) -> list[str]:
    """Flatten preferred-field values for presentation compatibility."""
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        if isinstance(value, str) and re.sub(r"\s+", " ", value.strip()).casefold() not in _GENERIC_FIELD_QUALIFIERS:
            result.append(value)
        elif isinstance(value, dict):
            result.extend(str(v) for v in value.values() if isinstance(v, str) and v.strip())
    return result


def normalize_jd(record: dict[str, Any], domain: str | None = None) -> dict[str, Any]:
    data = record.get("parsed_data") if isinstance(record.get("parsed_data"), dict) else record
    years = parse_experience(data.get("min_experience_years"))
    return {
        "id": record.get("id"), "domain": record.get("domain", domain),
        "role": {"job_title": data.get("job_title"), "overview": data.get("job_overview")},
        "skills": {"required": normalize_skills(data.get("required_skills")), "preferred": normalize_skills(data.get("preferred_skills"))},
        "experience": {"minimum_years": years},
        "education": {"minimum_degree": data.get("required_degree"), "preferred_fields": _atomic_fields(data.get("preferred_fields"))},
        "responsibilities": data.get("responsibilities") if isinstance(data.get("responsibilities"), list) else [],
        "certifications": data.get("required_certifications") if isinstance(data.get("required_certifications"), list) else [],
    }
