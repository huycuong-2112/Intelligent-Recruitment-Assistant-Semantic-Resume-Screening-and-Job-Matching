from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch

# optional imports; functions degrade gracefully
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

# rapidocr may be present in environment (Docling/rapidocr). Use if available.
try:
    import rapidocr
except Exception:
    rapidocr = None

class OCRFallback:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool | None = None) -> None:
        self.languages = languages or ["en"]
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
        self._easy_reader = None
        # Defer heavy/eager easyocr initialization; try lazy init on first use.

    def _easyocr_image(self, img_path: str) -> str:
        # attempt lazy init
        if self._easy_reader is None and easyocr is not None:
            try:
                self._easy_reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as e:
                raise RuntimeError(f"easyocr initialization failed: {e}")
        if self._easy_reader is None:
            raise RuntimeError("easyocr reader not available")
        results = self._easy_reader.readtext(img_path, detail=0)
        return "\n".join(results)

    def _pytesseract_image(self, img_path: str) -> str:
        if pytesseract is None or Image is None:
            raise RuntimeError("pytesseract or PIL not available")
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        return text

    def extract_from_image(self, img_path: str) -> str:
        last_exc = None
        # Try rapidocr (already present in environment used by Docling)
        if rapidocr is not None:
            try:
                # Try several common entrypoints dynamically
                for candidate in ("ocr", "read", "readtext", "main", "RapidOCR", "recognize", "run"):
                    if hasattr(rapidocr, candidate):
                        func = getattr(rapidocr, candidate)
                        if callable(func):
                            try:
                                # try simple call
                                out = func(img_path)
                                if isinstance(out, (list, tuple)):
                                    return "\n".join(str(x) for x in out)
                                return str(out)
                            except TypeError:
                                # try passing as keyword or different form
                                try:
                                    out = func([img_path])
                                    if isinstance(out, (list, tuple)):
                                        return "\n".join(str(x) for x in out)
                                    return str(out)
                                except Exception:
                                    raise
                # fallback: if rapidocr has a high-level runner
                if hasattr(rapidocr, "main") and callable(rapidocr.main):
                    try:
                        out = rapidocr.main(img_path)
                        return str(out)
                    except Exception:
                        pass
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
        raise RuntimeError(f"No OCR backend succeeded: {last_exc}")

    def extract(self, path: str) -> str:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return self.extract_from_image(path)
        try:
            from pdf2image import convert_from_path  # type: ignore
        except Exception:
            raise RuntimeError("PDF OCR requested but pdf2image is not installed; install pdf2image+poppler")
        pages = convert_from_path(path, dpi=300)
        texts = []
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                page.save(tmp.name, format="PNG")
                try:
                    texts.append(self.extract_from_image(tmp.name))
                finally:
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass
        return "\n\n".join(texts)
