from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite, sqrt
from typing import Any, Iterable

AVAILABLE = "available"
UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"

@dataclass
class ComponentResult:
    score: float | None
    availability: str
    status: str
    weight: float = 1.0
    details: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_components(components: Iterable[ComponentResult]) -> tuple[float | None, float]:
    items = list(components)
    applicable = [c for c in items if c.availability != NOT_APPLICABLE]
    denominator = sum(c.weight for c in applicable)
    evaluable = [c for c in applicable if c.availability == AVAILABLE and c.score is not None]
    if denominator <= 0:
        return None, 0.0
    coverage = sum(c.weight for c in evaluable) / denominator
    if not evaluable:
        return None, 0.0
    return sum(c.weight * float(c.score) for c in evaluable) / sum(c.weight for c in evaluable), coverage


def validate_vector(vector: Any) -> list[float]:
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("embedding vector must be a non-empty list")
    result = [float(x) for x in vector]
    if not all(isfinite(x) for x in result):
        raise ValueError("embedding vector contains NaN or infinity")
    return result


def cosine(a: Any, b: Any) -> float:
    left, right = validate_vector(a), validate_vector(b)
    if len(left) != len(right):
        raise ValueError("incompatible embedding dimensions")
    denominator = sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    if denominator == 0:
        raise ValueError("zero-norm embedding vector")
    return sum(x*y for x, y in zip(left, right)) / denominator
