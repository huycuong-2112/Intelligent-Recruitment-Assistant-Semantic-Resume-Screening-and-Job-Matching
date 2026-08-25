"""Build lightweight typed feature views; no vectors or semantic inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CVFeatures:
    id: str | None = None
    domain: str | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    skill_provenance: dict[str, list[str]] = field(default_factory=dict)
    professional_years: float | None = None
    education: dict[str, Any] = field(default_factory=dict)
    project_evidence: list[Any] = field(default_factory=list)
    work_evidence: list[Any] = field(default_factory=list)
    projects: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class JDFeatures:
    id: str | None = None
    domain: str | None = None
    role: dict[str, Any] = field(default_factory=dict)
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    minimum_years: float | None = None
    education: dict[str, Any] = field(default_factory=dict)
    responsibilities: list[Any] = field(default_factory=list)


def build_cv_features(normalized: dict[str, Any]) -> CVFeatures:
    skills = normalized.get("skills", {})
    return CVFeatures(
        id=normalized.get("id"), domain=normalized.get("domain"),
        profile=normalized.get("profile", {}), skills=skills.get("all", []),
        skill_provenance={"explicit": skills.get("explicit", []), "project_derived": skills.get("project_derived", []), "all": skills.get("all", [])},
        professional_years=normalized.get("experience", {}).get("professional_years"),
        education=normalized.get("education", {}),
        project_evidence=normalized.get("experience", {}).get("project_evidence", []),
        work_evidence=normalized.get("experience", {}).get("work_evidence", []),
        projects=normalized.get("projects", []),
    )


def build_jd_features(normalized: dict[str, Any]) -> JDFeatures:
    skills = normalized.get("skills", {})
    return JDFeatures(
        id=normalized.get("id"), domain=normalized.get("domain"), role=normalized.get("role", {}),
        required_skills=skills.get("required", []), preferred_skills=skills.get("preferred", []),
        minimum_years=normalized.get("experience", {}).get("minimum_years"),
        education=normalized.get("education", {}), responsibilities=normalized.get("responsibilities", []),
    )
