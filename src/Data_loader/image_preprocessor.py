from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS


def extract_document_cards(image_path: str) -> List[str]:
    """Phát hiện và cắt các trang tài liệu riêng biệt từ ảnh chụp viewer/screenshot."""
    p = Path(image_path)
    if not p.is_file():
        return [image_path]

    with Image.open(p) as raw_img:
        img_rgb = np.array(ImageOps.exif_transpose(raw_img).convert("RGB"))

    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Tạo mask lọc vùng giấy trắng sáng (nền CV thường > 220)
    _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_boxes = []
    min_page_area = (w * h) * 0.15

    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        area = w_c * h_c
        if area >= min_page_area and w_c > (w * 0.4):
            card_boxes.append((x_c, y_c, w_c, h_c))

    if not card_boxes:
        return [image_path]

    card_boxes.sort(key=lambda b: b[1])

    temp_paths = []
    for x_c, y_c, w_c, h_c in card_boxes:
        pad_x = min(4, w_c // 60)
        pad_y = min(4, h_c // 60)
        crop = img_rgb[y_c + pad_y : y_c + h_c - pad_y, x_c + pad_x : x_c + w_c - pad_x]

        if crop.size == 0:
            continue

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        Image.fromarray(crop).save(tmp_path, format="PNG")
        temp_paths.append(tmp_path)

    return temp_paths if temp_paths else [image_path]


def enhance_image(path: str, target_min_width: int = 1600) -> str:
    """Upscale nhẹ nhàng cho các ảnh quá nhỏ để OCR nhận diện tốt hơn."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")

    with Image.open(p) as img_raw:
        img = ImageOps.exif_transpose(img_raw).convert("RGB")

    w, h = img.size
    if w < target_min_width:
        scale = target_min_width / float(w)
        new_size = (target_min_width, int(h * scale))
        img = img.resize(new_size, resample=RESAMPLE_FILTER)

    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageOps.autocontrast(img, cutoff=1)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    img.save(tmp_path, format="PNG")
    return tmp_path


def cleanup_temp(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass