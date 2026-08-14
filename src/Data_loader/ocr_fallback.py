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


class OCRFallback:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool | None = None) -> None:
        self.languages = languages or ["en", "vi"]
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
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
            raise RuntimeError("RapidOCR engine not available")
        result, _ = engine(img_path)
        if not result:
            return ""
        # Extract text strings from RapidOCR tuple structure [[box, text, score], ...]
        return "\n".join(item[1] for item in result if len(item) > 1 and item[1])

    def _easyocr_image(self, img_path: str) -> str:
        if self._easy_reader is None and easyocr is not None:
            try:
                self._easy_reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as e:
                raise RuntimeError(f"EasyOCR init failed: {e}")
        if self._easy_reader is None:
            raise RuntimeError("EasyOCR reader not available")
        results = self._easy_reader.readtext(img_path, detail=0)
        return "\n".join(results)

    def _pytesseract_image(self, img_path: str) -> str:
        if pytesseract is None or Image is None:
            raise RuntimeError("pytesseract or PIL not available")
        img = Image.open(img_path)
        # Add support for English + Vietnamese and Preserve Layout (PSM 6/11)
        text = pytesseract.image_to_string(img, lang="eng+vie", config="--psm 11")
        return text

    def extract_from_image(self, img_path: str) -> str:
        last_exc = None

        if rapidocr_module is not None:
            try:
                return self._rapidocr_image(img_path)
            except Exception as exc:
                last_exc = exc

        if easyocr is not None:
            try:
                return self._easyocr_image(img_path)
            except Exception as exc:
                last_exc = exc

        if pytesseract is not None and Image is not None:
            try:
                return self._pytesseract_image(img_path)
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"No OCR engine succeeded. Last error: {last_exc}")

    def extract(self, path: str) -> str:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return self.extract_from_image(path)

        try:
            from pdf2image import convert_from_path
        except Exception:
            raise RuntimeError("PDF OCR requires pdf2image and poppler installed.")

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