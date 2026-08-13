from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Tuple

# Configuration / weights: dễ chỉnh ở một nơi.
QUALITY_THRESHOLD = 0.70

WEIGHTS = {
    "text_length": 0.25,
    "section_presence": 0.20,
    "abnormal_char": 0.20,
    "word_quality": 0.20,
    "repetition": 0.15,
}

COMMON_SECTION_HEADINGS = [
    "education", "experience", "work experience", "professional experience",
    "skills", "projects", "summary", "profile", "certifications",
    "achievements", "languages", "contact"
]

ABNORMAL_CHARS = set("|&<>^~@#$%*_=+`")
ABNORMAL_SEQUENCES = [r"(&gt;|&lt;|&amp;)", r"\|{2,}", r"[^\x00-\x7F]{6,}"]

def _normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def text_length_score(text: str) -> float:
    length = len(text)
    # ramp: small penalty for very short; saturate after 1500 chars
    if length <= 100:
        return 0.0
    if length >= 1500:
        return 1.0
    # scale between 100..1500
    return (length - 100) / (1500 - 100)

def section_presence_score(text: str) -> float:
    low = text.lower()
    found = sum(1 for h in COMMON_SECTION_HEADINGS if h in low)
    # presence / expectation
    return min(1.0, found / 4.0)

def abnormal_char_score(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    abnormal_count = sum(1 for ch in text if ch in ABNORMAL_CHARS)
    seq_penalty = 0
    for seq in ABNORMAL_SEQUENCES:
        seq_penalty += len(re.findall(seq, text))
    score = 1.0 - min(1.0, (abnormal_count / max(1, total)) + 0.1 * seq_penalty)
    return max(0.0, score)

def word_quality_score(text: str) -> float:
    tokens = re.findall(r"[A-Za-z]{2,}", text)
    if not tokens:
        return 0.0
    total_tokens = len(re.findall(r"\S+", text))
    alpha_tokens = len(tokens)
    ratio = alpha_tokens / max(1, total_tokens)
    return min(1.0, ratio)

def repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    most_common_ratio = counts.most_common(1)[0][1] / len(lines)
    # penalize if >30% of lines identical
    return max(0.0, 1.0 - most_common_ratio)

def evaluate(text: str) -> Tuple[float, Dict[str, float]]:
    t = _normalize_text(text)
    metrics = {}
    metrics["text_length"] = text_length_score(t)
    metrics["section_presence"] = section_presence_score(t)
    metrics["abnormal_char"] = abnormal_char_score(t)
    metrics["word_quality"] = word_quality_score(t)
    metrics["repetition"] = repetition_score(t)

    total = 0.0
    for k, w in WEIGHTS.items():
        total += metrics.get(k, 0.0) * w

    score = float(max(0.0, min(1.0, total)))
    return score, metrics

def is_pass(score: float, threshold: float | None = None) -> bool:
    if threshold is None:
        threshold = QUALITY_THRESHOLD
    return score >= threshold
