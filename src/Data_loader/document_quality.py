from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Tuple

# Điều chỉnh ngưỡng hợp lý (0.65) để tối ưu hiệu năng giữa Docling và OCR
QUALITY_THRESHOLD = 0.65

WEIGHTS = {
    "text_length": 0.20,
    "section_presence": 0.25,
    "abnormal_char": 0.20,
    "word_quality": 0.20,
    "repetition": 0.15,
}

# Danh mục tiêu đề mục (English + Vietnamese)
COMMON_SECTION_HEADINGS = [
    # English
    "education", "experience", "work experience", "professional experience",
    "skills", "projects", "summary", "profile", "certifications",
    "achievements", "languages", "contact",
    # Vietnamese
    "học vấn", "kinh nghiệm", "kinh nghiệm làm việc", "kỹ năng", 
    "dự án", "tóm tắt", "chứng chỉ", "thành tích", "ngôn ngữ", "liên hệ"
]

# Ký tự rác thực sự từ OCR (không tính ký tự định dạng Markdown/CV)
NOISY_CHARS = set("<>^~#$%`\\")


def _normalize_text(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = re.sub(r"[ \t]+", " ", t)
    return t


def text_length_score(text: str) -> float:
    length = len(text)
    if length <= 50:
        return 0.0
    if length >= 600:
        return 1.0
    return (length - 50) / (600 - 50)


def section_presence_score(text: str) -> float:
    low = text.lower()
    found_count = 0
    
    # Kiểm tra ranh giới từ để tránh đếm trùng lặp
    for heading in COMMON_SECTION_HEADINGS:
        if re.search(r'\b' + re.escape(heading) + r'\b', low):
            found_count += 1
            
    # Có từ 2-3 mục chuẩn là đạt điểm tuyệt đối
    return min(1.0, found_count / 2.5)


def abnormal_char_score(text: str) -> float:
    if not text:
        return 0.0
    
    # Loại bỏ cú pháp Markdown trước khi đo độ nhiễu
    stripped = re.sub(r"\||#|\*|_|-|:|•", "", text)
    total = len(stripped)
    if total == 0:
        return 1.0

    noisy_count = sum(1 for ch in stripped if ch in NOISY_CHARS)
    ratio = noisy_count / total
    return max(0.0, 1.0 - (ratio * 5.0))


def word_quality_score(text: str) -> float:
    # Lọc bỏ ký tự Markdown trước khi đếm token
    clean_text = re.sub(r"[\|\#\*\_\-\•\:\(\)\[\]]", " ", text)
    
    # Chấp nhận từ từ 1 ký tự trở lên (bao gồm C, R, v.v.)
    tokens = re.findall(r"[\wÀ-ỹ]{1,}", clean_text, flags=re.UNICODE)
    total_tokens = len(re.findall(r"\S+", clean_text))
    
    if not total_tokens:
        return 0.0
    
    ratio = len(tokens) / total_tokens
    return min(1.0, ratio)


def repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    if not lines:
        return 0.0
    
    # Nếu văn bản chỉ có 1-2 dòng dài hợp lệ thì không phạt lặp lại
    if len(lines) <= 2:
        return 1.0
        
    counts = Counter(lines)
    most_common_count = counts.most_common(1)[0][1]
    
    # Tỷ lệ lặp lại của dòng xuất hiện nhiều nhất
    most_common_ratio = most_common_count / len(lines)
    
    # Chỉ trừ điểm nặng nếu 1 dòng lặp lại quá 20% tổng số dòng
    if most_common_ratio <= 0.20:
        return 1.0
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