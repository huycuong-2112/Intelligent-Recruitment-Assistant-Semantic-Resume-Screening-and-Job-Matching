from __future__ import annotations

import os
import tempfile
from pathlib import Path
import torch

try:
    import easyocr
except Exception:
    easyocr = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import rapidocr_onnxruntime as rapidocr_module
except Exception:
    try:
        import rapidocr as rapidocr_module
    except Exception:
        rapidocr_module = None

try:
    from .document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass
except Exception:
    from document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass


class OCRFallback:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool | None = None) -> None:
        self.languages = languages or ["en", "vi"]
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self._easy_reader = None
        self._rapid_engine = None

    def _get_rapidocr_engine(self):
        if self._rapid_engine is None and rapidocr_module is not None:
            if hasattr(rapidocr_module, "RapidOCR"):
                self._rapid_engine = rapidocr_module.RapidOCR()
            elif callable(rapidocr_module):
                self._rapid_engine = rapidocr_module()
        return self._rapid_engine

    def _rapidocr_image(self, img_path: str) -> str:
        engine = self._get_rapidocr_engine()
        if engine is None:
            raise RuntimeError("RapidOCR engine unavailable")
        
        result, _ = engine(img_path)
        if not result:
            return ""

        # Sắp xếp các bounding box theo trục dọc (y) rồi đến trục ngang (x)
        sorted_boxes = sorted(result, key=lambda item: (round(item[0][0][1] / 20) * 20, item[0][0][0]))
        lines = []
        current_line = []
        last_y1 = -1

        for box, text, conf in sorted_boxes:
            if not text.strip():
                continue
            y1 = box[0][1]

            # Kiểm tra ngắt dòng thích ứng theo tỷ lệ linh hoạt
            if last_y1 != -1 and abs(y1 - last_y1) > 18:
                lines.append(" ".join(current_line))
                current_line = []

            current_line.append(text.strip())
            last_y1 = y1

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def _easyocr_image(self, img_path: str) -> str:
        if self._easy_reader is None and easyocr is not None:
            self._easy_reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        if self._easy_reader is None:
            raise RuntimeError("EasyOCR reader unavailable")
        results = self._easy_reader.readtext(img_path, detail=0, paragraph=True)
        return "\n".join(results)

    def _pytesseract_image(self, img_path: str) -> str:
        if pytesseract is None or Image is None:
            raise RuntimeError("pytesseract or PIL unavailable")
        # Sử dụng with để giải phóng file ngay lập tức, tránh lỗi PermissionError trên Windows
        with Image.open(img_path) as img:
            return pytesseract.image_to_string(img, lang="eng+vie", config="--psm 11")

    def extract_from_image(self, img_path: str) -> str:
        """
        Chiến lược tối ưu:
        1. Chạy RapidOCR đầu tiên (tốc độ cao).
        2. Nếu đạt chất lượng (Pass), trả về ngay.
        3. Nếu chưa đạt, chạy thêm EasyOCR / Tesseract để chấm điểm so tài.
        """
        candidates = []

        # 1. Chạy ưu tiên RapidOCR (ONNX cực nhanh)
        if rapidocr_module is not None:
            try:
                text_rapid = self._rapidocr_image(img_path)
                if text_rapid.strip():
                    score, _ = evaluate_quality(text_rapid)
                    candidates.append((score, text_rapid, "RapidOCR"))
                    # Nếu đạt chuẩn chất lượng -> Trả về luôn để tiết kiệm 90% thời gian
                    if quality_is_pass(score, threshold=0.70):
                        return text_rapid
            except Exception:
                pass

        # 2. Thử nghiệm dự phòng EasyOCR nếu RapidOCR chưa tối ưu
        if easyocr is not None:
            try:
                text_easy = self._easyocr_image(img_path)
                if text_easy.strip():
                    score, _ = evaluate_quality(text_easy)
                    candidates.append((score, text_easy, "EasyOCR"))
            except Exception:
                pass

        # 3. Thử nghiệm PyTesseract
        if pytesseract is not None and Image is not None:
            try:
                text_tess = self._pytesseract_image(img_path)
                if text_tess.strip():
                    score, _ = evaluate_quality(text_tess)
                    candidates.append((score, text_tess, "PyTesseract"))
            except Exception:
                pass

        if not candidates:
            raise RuntimeError("All OCR engines failed to extract text or returned empty results.")

        # Chọn kết quả có chất lượng cao nhất
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text, best_engine = candidates[0]
        return best_text

    def extract(self, path: str) -> str:
        p = Path(path)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return self.extract_from_image(path)

        try:
            from pdf2image import convert_from_path
        except Exception:
            raise RuntimeError("pdf2image is required for PDF OCR. Cài đặt: pip install pdf2image")

        # Đặt DPI = 200 để tối ưu tốc độ và bộ nhớ RAM
        pages = convert_from_path(str(p), dpi=200)
        texts = []
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_filename = tmp.name
                page.save(temp_filename, format="PNG")
            
            try:
                texts.append(self.extract_from_image(temp_filename))
            finally:
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except Exception:
                        pass
                        
        return "\n\n".join(texts)