from __future__ import annotations

import re
import unicodedata

# Toàn bộ phụ âm đơn và phụ âm ghép trong tiếng Việt (Không thể đứng độc lập)
ALL_VIETNAMESE_CONSONANTS = (
    r'(?:ngh|Ngh|NGH|ng|Ng|NG|nh|Nh|NH|ch|Ch|CH|th|Th|TH|tr|Tr|TR|'
    r'ph|Ph|PH|kh|Kh|KH|gh|Gh|GH|qu|Qu|QU|gi|Gi|GI|'
    r'[b-df-hj-np-tv-zđB-DF-HJ-NP-TV-ZĐ])'
)

# Từ điển chuẩn hóa các tiêu đề mục bị OCR làm méo dấu
HEADER_REPLACEMENTS = [
    (r'(?i)#*\s*hc\s*v[ãa]n\b', 'HỌC VẤN'),
    (r'(?i)#*\s*k[ýy]n[ăa]\s*ng\b', 'KỸ NĂNG'),
    (r'(?i)#*\s*k\s+n\s*ăng\b', 'KỸ NĂNG'),
    (r'(?i)#*\s*kinh\s+nghi\s*m\b', 'KINH NGHIỆM'),
    (r'(?i)#*\s*d\s+[áa]n\b', 'DỰ ÁN'),
    (r'(?i)#*\s*th\s*ng\s*tin\s*c\s*nh\s*n\b', 'THÔNG TIN CÁ NHÂN'),
]


def clean_logo_and_seal_noise(line: str) -> str:
    """Loại bỏ các dòng rác sinh ra từ con dấu tròn, logo mờ, watermark."""
    # Nếu dòng chứa chuỗi ký tự lặp vô nghĩa (VD: "og g gg", "ny gn ngnh")
    if re.search(r'\b([a-z])\s+\1\s+\1\b', line, re.IGNORECASE):
        return ""
    # Nếu dòng có quá nhiều ký hiệu lạ và chữ vô nghĩa
    if re.search(r'\(.*?\)[a-z\s]{1,4}:', line):
        return ""
    return line


def fix_glued_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Ép toàn bộ ký tự về chuẩn dựng sẵn (NFC)
    t = unicodedata.normalize("NFC", text)
    
    # 2. Xóa các tag comment của Docling
    t = re.sub(r'<!--\s*(?:image|table|figure|page break)\s*-->', '', t, flags=re.IGNORECASE)

    # 3. Nối các phụ âm tiếng Việt bị tách rời do OCR (qu ản -> quản, tr ị -> trị, t ôi -> tôi, c ơ -> cơ)
    consonants = (
        r'(?:ngh|Ngh|NGH|ng|Ng|NG|nh|Nh|NH|ch|Ch|CH|th|Th|TH|tr|Tr|TR|'
        r'ph|Ph|PH|kh|Kh|KH|gh|Gh|GH|qu|Qu|QU|gi|Gi|GI|'
        r'[b-df-hj-np-tv-zđB-DF-HJ-NP-TV-ZĐ])'
    )
    for _ in range(2):
        t = re.sub(rf'\b({consonants})\s+([a-zA-Zà-ỹÀ-Ỹ0-9]+)\b', r'\1\2', t)

    # 4. Hàn phụ âm cuối bị ngắt (VIỆ T -> VIỆT, NĂ NG -> NĂNG)
    broken_suffixes = r'(?:[cCntTmMpP]|ng|nh)'
    t = re.sub(rf'\b([a-zA-Zà-ỹÀ-Ỹ]{{2,}}[áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĨịóòỏõọốồổỗỘớờởỡợúùủũụứừửữỰýỳỷỹỵÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ])\s+({broken_suffixes})\b', r'\1\2', t)

    # 5. Khôi phục khoảng trắng dính từ & nén dòng trống
    t = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", t)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    return "\n".join([line for line in lines if line])