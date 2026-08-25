"""Normalize parsed job-description records deterministically."""

from __future__ import annotations

from typing import Any

from .skill_normalizer import normalize_skills
from .experience import parse_experience


def normalize_jd(record: dict[str, Any], domain: str | None = None) -> dict[str, Any]:
    data = record.get("parsed_data") if isinstance(record.get("parsed_data"), dict) else record
    years = parse_experience(data.get("min_experience_years"))
    return {
        "id": record.get("id"), "domain": record.get("domain", domain),
        "role": {"job_title": data.get("job_title"), "overview": data.get("job_overview")},
        "skills": {"required": normalize_skills(data.get("required_skills")), "preferred": normalize_skills(data.get("preferred_skills"))},
        "experience": {"minimum_years": years},
        "education": {"minimum_degree": data.get("required_degree"), "preferred_fields": data.get("preferred_fields") if isinstance(data.get("preferred_fields"), list) else []},
        "responsibilities": data.get("responsibilities") if isinstance(data.get("responsibilities"), list) else [],
        "certifications": data.get("required_certifications") if isinstance(data.get("required_certifications"), list) else [],
    }
