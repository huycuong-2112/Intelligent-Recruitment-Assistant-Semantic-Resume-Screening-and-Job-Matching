from __future__ import annotations
import re

def fix_glued_text(text: str) -> str:
    """
    Khôi phục khoảng trắng bị thiếu (Glued text) 
    và hàn gắn các âm tiết tiếng Việt bị đứt gãy do OCR (Over-segmentation).
    """
    if not text:
        return ""

    t = text

    # =========================================================
    # BƯỚC 1: HÀN CHỮ BỊ ĐỨT DO OCR (OVER-SEGMENTATION)
    # Xử lý trước để ghép các từ bị nát thành từ có nghĩa
    # =========================================================
    
    # 1. Chữa chữ in hoa tiêu đề bị đứt (VD: "TÓM TẮ T" -> "TÓM TẮT", "KỸ NĂ NG" -> "KỸ NĂNG")
    t = re.sub(r'\b([A-ZÀ-Ỹ]+) ([A-ZÀ-Ỹ]{1,2})\b', r'\1\2', t)

    # 2. Nối phụ âm kép/ba bị tách khỏi vần (VD: "qu ản" -> "quản", "tr ị" -> "trị", "ng ười" -> "người")
    t = re.sub(r'\b([qQ][uU]|[tT]r|[tT]h|[pP]h|[nN]h|[cC]h|[gG]i|[kK]h|[nN]g|[nN]gh) ([a-zA-Zà-ỹÀ-Ỹ]+)\b', r'\1\2', t)
    
    # 3. Nối các chữ cái đơn lẻ bị tách khỏi vần (VD: "n ăm" -> "năm", "h ệ" -> "hệ", "m ạng" -> "mạng", "c ố" -> "cố")
    t = re.sub(r'\b([a-zA-Zà-ỹÀ-Ỹ]) ([a-zA-Zà-ỹÀ-Ỹ]+)\b', r'\1\2', t)


    # =========================================================
    # BƯỚC 2: TÁCH CHỮ BỊ DÍNH VÀO NHAU (GLUED TEXT)
    # Khôi phục khoảng trắng tại các ranh giới ký tự
    # =========================================================
    
    # 1. Chữ thường dính liền chữ hoa (VD: "NetworkAdmin" -> "Network Admin")
    t = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", t)
    
    # 2. Chữ cái dính liền số (VD: "năm2025" -> "năm 2025")
    t = re.sub(r"([a-zA-Zà-ỹÀ-Ỹ])(\d)", r"\1 \2", t)
    
    # 3. Số dính liền chữ cái (VD: "2023hiện" -> "2023 hiện")
    t = re.sub(r"(\d)([a-zA-Zà-ỹÀ-Ỹ])", r"\1 \2", t)
    
    # 4. Tách dấu gạch đầu dòng, gạch chéo, gạch ngang bị dính vào chữ (VD: "•Quản" -> "• Quản")
    t = re.sub(r"([a-zA-Zà-ỹÀ-Ỹ0-9])([•|\-/—])", r"\1 \2", t)
    t = re.sub(r"([•|\-/—])([a-zA-Zà-ỹÀ-Ỹ0-9])", r"\1 \2", t)


    # =========================================================
    # BƯỚC 3: DỌN DẸP KHOẢNG TRẮNG & DÒNG TRỐNG
    # =========================================================
    
    # Xóa khoảng trắng thừa (nhiều space liên tiếp thành 1 space)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    
    # Lọc bỏ hoàn toàn các dòng trống (tiết kiệm token cho LLM)
    cleaned_lines = [line for line in lines if line]

    return "\n".join(cleaned_lines)