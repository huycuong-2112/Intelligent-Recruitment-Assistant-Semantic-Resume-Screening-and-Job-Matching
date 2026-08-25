"""Deterministic skill cleanup and canonicalization."""

from __future__ import annotations

import html
import re
from typing import Iterable

from .taxonomy import SKILL_ALIASES, alias_key


def normalize_skill(skill: str | None) -> str:
    if not isinstance(skill, str):
        return ""
    value = html.unescape(skill).strip().strip("•-*\t\n ")
    value = re.sub(r"\s+", " ", value)
    if not value:
        return ""
    return SKILL_ALIASES.get(alias_key(value), value)


def normalize_skills(skills: Iterable[str] | None) -> list[str]:
    if not skills:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in skills:
        normalized = normalize_skill(item)
        key = alias_key(normalized)
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result
