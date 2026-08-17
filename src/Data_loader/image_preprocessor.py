from __future__ import annotations

import os
import tempfile
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# Tương thích giữa các phiên bản Pillow cũ và mới
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS


def enhance_image(path: str, scale: float = 1.0) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")

    # Đọc và giải phóng file gốc ngay lập tức để tránh khóa file trên Windows
    with Image.open(p) as img_raw:
        img = img_raw.convert("RGB")

    # 1. Tăng độ tương phản để làm nổi bật chữ đen trên nền trắng
    img = ImageEnhance.Contrast(img).enhance(1.4)

    # 2. Tăng độ sắc nét của các nét chữ
    img = ImageEnhance.Sharpness(img).enhance(1.3)
    img = img.filter(ImageFilter.SHARPEN)

    # 3. Phóng to ảnh (nếu cần) với bộ lọc Lanczos chất lượng cao
    if scale != 1.0 and scale > 0:
        w, h = img.size
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, resample=RESAMPLE_FILTER)

    # Tạo file tạm an toàn cho Windows (đóng file handle trước khi lưu ảnh)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    img.save(tmp_path, format="PNG")
    return tmp_path


def cleanup_temp(path: str) -> None:
    """Xóa file ảnh tạm sau khi xử lý xong OCR."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass