from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Tuple

QUALITY_THRESHOLD = 0.70

WEIGHTS = {
    "text_length": 0.10,
    "section_presence": 0.15,
    "fragmentation": 0.25,        # Bắt lỗi tách âm tiết (t ôi, c ơ, kh í, k n ăng)
    "vowel_validity": 0.25,       # Bắt lỗi rụng nguyên âm (nn, tng, vt, trng, nm, thn)
    "vietnamese_diacritic": 0.15, # Bắt lỗi rụng dấu tiếng Việt
    "abnormal_char": 0.10,        # Bắt HTML artifact & ký tự rác
    "repetition": 0.05,
}

COMMON_SECTION_HEADINGS = [
    # English
    "education", "experience", "work experience", "professional experience",
    "skills", "projects", "summary", "profile", "certifications",
    # Vietnamese
    "học vấn", "kinh nghiệm", "kỹ năng", "dự án", "tóm tắt", "chứng chỉ", "thành tích",
    # Mutilated / Unaccented
    "hoc van", "kinh nghiem", "ky nang", "du an", "hc van", "kynang"
]

# Nhận diện tiếng Việt dựa trên ký tự độc bản hoặc từ vựng/địa danh phổ biến
VN_UNIQUE_CHARS = re.compile(r'[đĐưƯơƠăĂâÂêÊôÔáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]', re.IGNORECASE)
VN_LEXICAL_MARKERS = re.compile(
    r'\b(nguyen|tran|le|pham|hoang|huynh|phan|vu|vo|dang|bui|do|ho|ngo|duong|ly|'
    r'hcm|tp\.hcm|hcmut|hcmute|hanoi|vietnam|viet\s*nam|sinh\s*vien|dai\s*hoc|'
    r'tphcm|quan|phuong|thanh\s*pho|cong\s*ty|trach\s*nhiem|chuyen\s*nganh)\b',
    re.IGNORECASE
)

VALID_VOWELS = set("aeiouyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵAEIOUYÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ")
VALID_SINGLE_LETTERS = set("aiAIyY")  # Các chữ đơn lẻ hợp lệ trong tiếng Anh/Việt


def _normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").strip()
    return re.sub(r"[ \t]+", " ", t)


def text_length_score(text: str) -> float:
    length = len(text)
    if length <= 50:
        return 0.0
    if length >= 500:
        return 1.0
    return (length - 50) / (500 - 50)


def section_presence_score(text: str) -> float:
    low = text.lower()
    found_count = sum(1 for h in COMMON_SECTION_HEADINGS if re.search(r'\b' + re.escape(h) + r'\b', low))
    return min(1.0, found_count / 2.0)


def fragmentation_score(text: str) -> float:
    """
    Phát hiện chữ bị tách rời thành từng ký tự lẻ do OCR (VD: 't ôi', 'b án', 'k n ăng', 'c ơ kh í').
    """
    words = re.findall(r'\b[a-zA-Zà-ỹÀ-ỸđĐ0-9]+\b', text)
    if not words:
        return 0.0

    single_letter_count = sum(1 for w in words if len(w) == 1 and w not in VALID_SINGLE_LETTERS and not w.isdigit())
    frag_ratio = single_letter_count / len(words)

    # Nếu tỷ lệ chữ 1 ký tự > 12%, trừ điểm nặng
    if frag_ratio <= 0.02:
        return 1.0
    elif frag_ratio >= 0.15:
        return 0.0
    return 1.0 - ((frag_ratio - 0.02) / (0.15 - 0.02))


def vowel_validity_score(text: str) -> float:
    """
    Phát hiện các từ vô nghĩa không có nguyên âm do font hỏng (VD: 'nn', 'tng', 'vt', 'trng', 'nm', 'thn', 'mc').
    Mỗi từ hợp lệ (độ dài >= 2) trong tiếng Anh và tiếng Việt bắt buộc phải chứa ít nhất 1 nguyên âm.
    """
    words = re.findall(r'\b[a-zA-Zà-ỹÀ-ỸđĐ]+\b', text)
    multiletter_words = [w for w in words if len(w) >= 2]
    if not multiletter_words:
        return 0.0

    vowelless_count = 0
    for w in multiletter_words:
        if not any(ch in VALID_VOWELS for ch in w):
            vowelless_count += 1

    vowelless_ratio = vowelless_count / len(multiletter_words)

    # Tỷ lệ từ không nguyên âm > 3% là dấu hiệu văn bản bị nát
    if vowelless_ratio <= 0.01:
        return 1.0
    elif vowelless_ratio >= 0.06:
        return 0.0
    return 1.0 - ((vowelless_ratio - 0.01) / (0.06 - 0.01))


def vietnamese_diacritic_score(text: str) -> float:
    """
    Phát hiện tài liệu gốc là tiếng Việt nhưng bị rụng dấu thanh.
    """
    low = text.lower()
    has_vn_chars = bool(VN_UNIQUE_CHARS.search(text))
    has_vn_lexicon = bool(VN_LEXICAL_MARKERS.search(low))

    # Nếu hoàn toàn không có dấu hiệu tiếng Việt -> Coi là tiếng Anh thuần
    if not has_vn_chars and not has_vn_lexicon:
        return 1.0

    diacritics_pattern = re.compile(r'[à-ỹÀ-ỸđĐ]')
    diacritic_chars = diacritics_pattern.findall(text)
    
    words = [w for w in re.findall(r'\b[a-zA-Zà-ỹÀ-ỸđĐ]+\b', text) if len(w) > 1]
    if not words:
        return 0.0

    # Văn bản tiếng Việt chuẩn có mật độ ký tự có dấu >= 25% tổng số từ
    diacritic_ratio = len(diacritic_chars) / len(words)
    if diacritic_ratio >= 0.25:
        return 1.0
    elif diacritic_ratio <= 0.05:
        return 0.1  # Bị rụng dấu gần hết
    return diacritic_ratio / 0.25


def abnormal_char_score(text: str) -> float:
    if not text:
        return 0.0
    
    # Phạt nếu chứa HTML tags từ Docling hoặc chuỗi logo lặp vô nghĩa
    html_comments = len(re.findall(r'<!--.*?-->', text))
    logo_noise = len(re.findall(r'\b([a-z])\s+\1\s+\1\b', text, re.IGNORECASE))
    
    clean = re.sub(r"[\|\#\*\_\-\•\:\(\)\[\]\n\s]", "", text)
    if not clean:
        return 0.0

    noisy_chars = sum(1 for ch in clean if ch in set("<>^~#$%`\\{}"))
    total_penalty = (noisy_chars * 3) + (html_comments * 20) + (logo_noise * 15)
    
    ratio = total_penalty / len(clean)
    return max(0.0, 1.0 - (ratio * 10.0))


def repetition_score(text: str) -> float:
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 3]
    if not lines or len(lines) <= 2:
        return 1.0
    counts = Counter(lines)
    most_common_ratio = counts.most_common(1)[0][1] / len(lines)
    return 1.0 if most_common_ratio <= 0.20 else max(0.0, 1.0 - most_common_ratio)


def evaluate(text: str) -> Tuple[float, Dict[str, float]]:
    t = _normalize_text(text)
    metrics = {
        "text_length": text_length_score(t),
        "section_presence": section_presence_score(t),
        "fragmentation": fragmentation_score(t),
        "vowel_validity": vowel_validity_score(t),
        "vietnamese_diacritic": vietnamese_diacritic_score(t),
        "abnormal_char": abnormal_char_score(t),
        "repetition": repetition_score(t),
    }

    total = sum(metrics[k] * w for k, w in WEIGHTS.items())
    score = float(max(0.0, min(1.0, total)))
    return score, metrics


def is_pass(score: float, threshold: float | None = None) -> bool:
    if threshold is None:
        threshold = QUALITY_THRESHOLD
    return score >= threshold