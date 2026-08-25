"""Parsing helpers that preserve unknown experience as ``None``."""

from __future__ import annotations

import re
from typing import Any


def parse_experience(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value)) if value >= 0 else None
    if isinstance(value, str):
        text = value.strip().casefold()
        if not text:
            return None
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|năm)?", text)
        if match:
            return float(match.group(1))
    return None
