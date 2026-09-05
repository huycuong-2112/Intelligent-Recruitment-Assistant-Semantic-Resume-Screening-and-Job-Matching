from __future__ import annotations

import concurrent.futures
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

logging.getLogger("RapidOCR").setLevel(logging.ERROR)

try:
    import easyocr
except Exception:
    easyocr = None

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

from document_quality import evaluate as evaluate_quality
from image_preprocessor import extract_document_cards, cleanup_temp


class OCRFallback:
    def __init__(self, languages: list[str] | None = None, use_gpu: bool | None = None) -> None:
        self.languages = languages or ["en", "vi"]
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self._easy_reader = None
        self._rapid_engine = None

    def _get_rapidocr_engine(self):
        if self._rapid_engine is None and rapidocr_module is not None:
            rapid_kwargs = {
                "det_limit_side_len": 960,  # Chuẩn tối ưu cho DBNet receptive field
                "det_db_thresh": 0.2,
                "det_db_box_thresh": 0.25,
                "det_db_unclip_ratio": 1.6,
            }
            try:
                if hasattr(rapidocr_module, "RapidOCR"):
                    self._rapid_engine = rapidocr_module.RapidOCR(**rapid_kwargs)
                elif callable(rapidocr_module):
                    self._rapid_engine = rapidocr_module(**rapid_kwargs)
            except Exception:
                try:
                    if hasattr(rapidocr_module, "RapidOCR"):
                        self._rapid_engine = rapidocr_module.RapidOCR()
                    elif callable(rapidocr_module):
                        self._rapid_engine = rapidocr_module()
                except Exception:
                    self._rapid_engine = None
        return self._rapid_engine

    def _sort_and_cluster_boxes(self, boxes_with_text: List[Tuple[list, str, float]]) -> str:
        if not boxes_with_text:
            return ""

        items = []
        for box, text, conf in boxes_with_text:
            if not text or not text.strip():
                continue
            y_coords = [p[1] for p in box]
            x_coords = [p[0] for p in box]
            items.append({
                "text": text.strip(),
                "x": min(x_coords),
                "y": sum(y_coords) / len(y_coords),
                "h": max(max(y_coords) - min(y_coords), 10),
            })

        if not items:
            return ""

        items.sort(key=lambda item: item["y"])
        lines = []
        current_line = [items[0]]

        for item in items[1:]:
            prev_item = current_line[-1]
            line_height = max(prev_item["h"], item["h"])
            if abs(item["y"] - prev_item["y"]) < (line_height * 0.7):
                current_line.append(item)
            else:
                current_line.sort(key=lambda x: x["x"])
                lines.append(" ".join(t["text"] for t in current_line))
                current_line = [item]

        if current_line:
            current_line.sort(key=lambda x: x["x"])
            lines.append(" ".join(t["text"] for t in current_line))

        return "\n".join(lines)

    def _run_rapidocr(self, img_bgr: np.ndarray) -> Tuple[str, str]:
        engine = self._get_rapidocr_engine()
        if engine is None:
            return "", "RapidOCR"
        try:
            result, _ = engine(img_bgr)
            if result:
                return self._sort_and_cluster_boxes(result), "RapidOCR"

            # Tiling Fallback nếu ảnh dọc dài
            h, w, _ = img_bgr.shape
            if h > w * 1.2:
                mid = h // 2
                overlap = int(h * 0.05)
                top_img = img_bgr[: mid + overlap, :]
                bot_img = img_bgr[mid - overlap :, :]

                res_top, _ = engine(top_img)
                res_bot, _ = engine(bot_img)

                combined = []
                if res_top:
                    combined.extend(res_top)
                if res_bot:
                    for b in res_bot:
                        box_pts = [[pt[0], pt[1] + (mid - overlap)] for pt in b[0]]
                        combined.append((box_pts, b[1], b[2]))

                if combined:
                    return self._sort_and_cluster_boxes(combined), "RapidOCR_Tiled"

            return "", "RapidOCR"
        except Exception:
            return "", "RapidOCR"

    def _run_easyocr(self, img_bgr: np.ndarray) -> Tuple[str, str]:
        if self._easy_reader is None and easyocr is not None:
            try:
                self._easy_reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception:
                return "", "EasyOCR"
        if self._easy_reader is None:
            return "", "EasyOCR"
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self._easy_reader.readtext(img_rgb, detail=0, paragraph=True)
            return "\n".join(results), "EasyOCR"
        except Exception:
            return "", "EasyOCR"

    def _run_pytesseract(self, img_bgr: np.ndarray) -> Tuple[str, str]:
        if pytesseract is None:
            return "", "PyTesseract"
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # Ưu tiên vie đứng trước eng để match đúng từ điển dấu tiếng Việt
            return pytesseract.image_to_string(img_rgb, lang="vie+eng", config="--psm 4"), "PyTesseract"
        except Exception:
            return "", "PyTesseract"

    def extract_single_card(self, img_path: str) -> str:
        with Image.open(img_path) as raw:
            rgb_arr = np.array(ImageOps.exif_transpose(raw).convert("RGB"))
            img_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        candidates: List[Tuple[float, str, str]] = []
        tasks = [
            (self._run_rapidocr, img_bgr),
            (self._run_easyocr, img_bgr),
            (self._run_pytesseract, img_bgr),
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(fn, arg) for fn, arg in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    text, engine_name = future.result()
                    if text and text.strip():
                        score, _ = evaluate_quality(text)
                        candidates.append((score, text, engine_name))
                except Exception:
                    continue

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def extract(self, path: str) -> str:
        p = Path(path)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            card_paths = extract_document_cards(path)
            extracted_pages = []

            for card in card_paths:
                text = self.extract_single_card(card)
                if text.strip():
                    extracted_pages.append(text)
                if card != path:
                    cleanup_temp(card)

            return "\n\n".join(extracted_pages)

        try:
            from pdf2image import convert_from_path
        except Exception:
            raise RuntimeError("pdf2image is required for PDF OCR: pip install pdf2image")

        pages = convert_from_path(str(p), dpi=300)
        texts = []
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_filename = tmp.name
                page.save(temp_filename, format="PNG")

            try:
                extracted = self.extract_single_card(temp_filename)
                if extracted.strip():
                    texts.append(extracted)
            finally:
                cleanup_temp(temp_filename)

        return "\n\n".join(texts)