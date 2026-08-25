"""Normalize parsed CV records while retaining evidence provenance."""

from __future__ import annotations

from typing import Any

from .skill_normalizer import normalize_skills
from .experience import parse_experience


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def normalize_cv(record: dict[str, Any], domain: str | None = None) -> dict[str, Any]:
    data = record.get("parsed_data") if isinstance(record.get("parsed_data"), dict) else record
    explicit = normalize_skills(data.get("skills"))
    projects = _as_list(data.get("projects"))
    project_derived: list[str] = []
    project_evidence: list[Any] = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        project_derived.extend(normalize_skills(project.get("technologies")))
        description = project.get("description") or project.get("summary") or project.get("details")
        if description is not None:
            project_evidence.append(description)
    project_derived = normalize_skills(project_derived)
    all_skills = normalize_skills(explicit + project_derived)
    work = _as_list(data.get("work_experience"))
    professional_years = parse_experience(data.get("experience_years"))
    return {
        "id": record.get("id"), "domain": record.get("domain", domain),
        "profile": {"summary": _text(data.get("summary")), "job_titles": normalize_skills(data.get("job_titles"))},
        "skills": {"all": all_skills, "explicit": explicit, "project_derived": project_derived},
        "experience": {"professional_years": professional_years, "work_evidence": work, "project_evidence": project_evidence},
        "education": {"degree": data.get("education_degree"), "field": data.get("education_field"), "history": _as_list(data.get("education_history"))},
        "projects": projects, "certifications": normalize_skills(data.get("certifications")),
    }
