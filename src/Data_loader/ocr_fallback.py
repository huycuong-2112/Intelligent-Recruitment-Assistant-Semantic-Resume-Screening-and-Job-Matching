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

# Import hàm đánh giá chất lượng để chấm điểm từng engine OCR
try:
    from .document_quality import evaluate as evaluate_quality
except Exception:
    from document_quality import evaluate as evaluate_quality


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

        lines = []
        sorted_boxes = sorted(result, key=lambda item: (round(item[0][0][1] / 10) * 10, item[0][0][0]))
        current_line = []
        last_x2 = -1
        last_y1 = -1

        for box, text, conf in sorted_boxes:
            x1, y1 = box[0][0], box[0][1]
            x2 = box[1][0]

            if not text.strip():
                continue

            if last_y1 != -1 and abs(y1 - last_y1) > 12:
                lines.append(" ".join(current_line))
                current_line = []
                last_x2 = -1

            if last_x2 != -1 and (x1 - last_x2) > 5:
                current_line.append(text.strip())
            else:
                if current_line:
                    current_line[-1] += f" {text.strip()}"
                else:
                    current_line.append(text.strip())

            last_x2 = x2
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
        img = Image.open(img_path)
        return pytesseract.image_to_string(img, lang="eng+vie", config="--psm 11")

    def extract_from_image(self, img_path: str) -> str:
        """
        Chiến lược mới: Chạy tất cả các engine OCR có sẵn, 
        chấm điểm từng kết quả và trả về văn bản có điểm chất lượng cao nhất.
        """
        candidates = []

        # 1. Thử RapidOCR
        if rapidocr_module is not None:
            try:
                text_rapid = self._rapidocr_image(img_path)
                if text_rapid.strip():
                    score, _ = evaluate_quality(text_rapid)
                    candidates.append((score, text_rapid, "RapidOCR"))
            except Exception:
                pass

        # 2. Thử EasyOCR
        if easyocr is not None:
            try:
                text_easy = self._easyocr_image(img_path)
                if text_easy.strip():
                    score, _ = evaluate_quality(text_easy)
                    candidates.append((score, text_easy, "EasyOCR"))
            except Exception:
                pass

        # 3. Thử PyTesseract
        if pytesseract is not None and Image is not None:
            try:
                text_tess = self._pytesseract_image(img_path)
                if text_tess.strip():
                    score, _ = evaluate_quality(text_tess)
                    candidates.append((score, text_tess, "PyTesseract"))
            except Exception:
                pass

        # Nếu không có engine nào chạy ra chữ
        if not candidates:
            raise RuntimeError("All OCR engines failed to extract any text or returned empty results.")

        # Sắp xếp các ứng viên theo điểm số giảm dần (score cao nhất đứng đầu)
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text, best_engine = candidates[0]

        print(f"   [OCR Competition] Winner: {best_engine} with score: {best_score:.3f}")
        return best_text

    def extract(self, path: str) -> str:
        p = Path(path)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return self.extract_from_image(path)

        try:
            from pdf2image import convert_from_path
        except Exception:
            raise RuntimeError("pdf2image is required for PDF OCR.")

        pages = convert_from_path(path, dpi=300)
        texts = []
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                page.save(tmp.name, format="PNG")
                try:
                    texts.append(self.extract_from_image(tmp.name))
                finally:
                    if os.path.exists(tmp.name):
                        os.remove(tmp.name)
        return "\n\n".join(texts)