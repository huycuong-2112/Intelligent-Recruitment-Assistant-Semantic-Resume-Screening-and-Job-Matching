from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Tuple

QUALITY_THRESHOLD = 0.80

WEIGHTS = {
    "text_length": 0.20,
    "section_presence": 0.25,
    "abnormal_char": 0.20,
    "word_quality": 0.20,
    "repetition": 0.15,
}

# Supported Multilingual Headings (English + Vietnamese)
COMMON_SECTION_HEADINGS = [
    # English
    "education", "experience", "work experience", "professional experience",
    "skills", "projects", "summary", "profile", "certifications",
    "achievements", "languages", "contact",
    # Vietnamese
    "học vấn", "kinh nghiệm", "kinh nghiệm làm việc", "kỹ năng", 
    "dự án", "tóm tắt", "chứng chỉ", "thành tích", "ngôn ngữ", "liên hệ"
]

# Characters that indicate genuine OCR noise, excluding standard Markdown/Resume formatting
NOISY_CHARS = set("<>^~#$%`\\")

def _normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"[ \t]+", " ", t)
    return t

def text_length_score(text: str) -> float:
    length = len(text)
    if length <= 50:
        return 0.0
    if length >= 800:
        return 1.0
    return (length - 50) / (800 - 50)

def section_presence_score(text: str) -> float:
    low = text.lower()
    found = sum(1 for h in COMMON_SECTION_HEADINGS if h in low)
    return min(1.0, found / 3.0)

def abnormal_char_score(text: str) -> float:
    if not text:
        return 0.0
    
    # Strip markdown syntax before evaluating noise
    stripped = re.sub(r"\||#|\*|_", "", text)
    total = len(stripped)
    if total == 0:
        return 1.0

    noisy_count = sum(1 for ch in stripped if ch in NOISY_CHARS)
    ratio = noisy_count / total
    return max(0.0, 1.0 - (ratio * 5.0))

def word_quality_score(text: str) -> float:
    # Unicode-aware word tokenization (supports Vietnamese and international alphabets)
    tokens = re.findall(r"[\w]{2,}", text, flags=re.UNICODE)
    total_tokens = len(re.findall(r"\S+", text))
    if not total_tokens:
        return 0.0
    
    ratio = len(tokens) / total_tokens
    return min(1.0, ratio)

def repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    most_common_ratio = counts.most_common(1)[0][1] / len(lines)
    return max(0.0, 1.0 - most_common_ratio)

def evaluate(text: str) -> Tuple[float, Dict[str, float]]:
    t = _normalize_text(text)
    metrics = {
        "text_length": text_length_score(t),
        "section_presence": section_presence_score(t),
        "abnormal_char": abnormal_char_score(t),
        "word_quality": word_quality_score(t),
        "repetition": repetition_score(t),
    }

    total = sum(metrics[k] * w for k, w in WEIGHTS.items())
    score = float(max(0.0, min(1.0, total)))
    return score, metrics

def is_pass(score: float, threshold: float | None = None) -> bool:
    if threshold is None:
        threshold = QUALITY_THRESHOLD
    return score >= threshold