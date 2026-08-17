from __future__ import annotations
import re


def fix_glued_text(text: str) -> str:
    """
    Khôi phục khoảng trắng bị thiếu (Glued text) 
    và hàn gắn các âm tiết tiếng Việt bị đứt gãy do OCR một cách an toàn.
    """
    if not text:
        return ""

    t = text

    # =========================================================
    # BƯỚC 1: HÀN CHỮ BỊ ĐỨT DO OCR (AN TOÀN - KHÔNG PHÁ HỦY TỪ ĐƠN)
    # =========================================================

    # 1. Nối các phụ âm ghép tiếng Việt KHÔNG THỂ đứng một mình khi bị tách khỏi vần
    # (VD: "qu ản" -> "quản", "tr ị" -> "trị", "ngh iệp" -> "nghiệp", "kh óa" -> "khóa")
    consonants = r'(?:ngh|Ngh|NGH|ng|Ng|NG|nh|Nh|NH|ch|Ch|CH|th|Th|TH|tr|Tr|TR|ph|Ph|PH|kh|Kh|KH|gh|Gh|GH|qu|Qu|QU)'
    t = re.sub(rf'\b({consonants})\s+([a-zA-Zà-ỹÀ-Ỹ]+)\b', r'\1\2', t)

    # 2. Hàn các nguyên âm/dấu thanh tiếng Việt bị OCR tách riêng lẻ ở cuối từ
    # (VD: "TẮ T" -> "TẮT", "TIẾ P" -> "TIẾP", "NĂ NG" -> "NĂNG", "VIỆ T" -> "VIỆT")
    broken_suffixes = r'(?:[cCntTmMpP]|ng|nh)'
    t = re.sub(rf'\b([A-ZÀ-Ỹ]{{2,}}[ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ])\s+({broken_suffixes})\b', r'\1\2', t)
    t = re.sub(rf'\b([a-zà-ỹ]{{2,}}[áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ])\s+({broken_suffixes})\b', r'\1\2', t)


    # =========================================================
    # BƯỚC 2: TÁCH CHỮ BỊ DÍNH VÀO NHAU (GLUED TEXT)
    # =========================================================

    # 1. Chữ thường dính liền chữ hoa (CamelCase hoặc lỗi dính từ)
    # (VD: "SoftwareEngineer" -> "Software Engineer", "quảnLý" -> "quản Lý")
    t = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", t)

    # 2. Tách dấu gạch đầu dòng / bullet point bị dính vào chữ
    # (VD: "•Quản lý" -> "• Quản lý", "*Python" -> "* Python")
    t = re.sub(r"([•▪\*\-—])([a-zA-Zà-ỹÀ-Ỹ0-9])", r"\1 \2", t)
    t = re.sub(r"([a-zA-Zà-ỹÀ-Ỹ0-9])([•▪])", r"\1 \2", t)

    # 3. Tách mốc năm dính liền từ tiếng Việt
    # (VD: "năm2024" -> "năm 2024", "2020đến" -> "2020 đến")
    t = re.sub(r"([a-zA-Zà-ỹÀ-Ỹ]{3,})(\d{4})", r"\1 \2", t)
    t = re.sub(r"(\d{4})([a-zA-Zà-ỹÀ-Ỹ]{3,})", r"\1 \2", t)


    # =========================================================
    # BƯỚC 3: CHUẨN HÓA KHOẢNG TRẮNG VÀ DÒNG TRỐNG
    # =========================================================

    # Gộp nhiều dấu cách ngang thành 1
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    
    # Loại bỏ dòng rỗng hoàn toàn để tinh giản token cho LLM
    cleaned_lines = [line for line in lines if line]

    return "\n".join(cleaned_lines)