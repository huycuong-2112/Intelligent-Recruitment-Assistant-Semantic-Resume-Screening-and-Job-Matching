from __future__ import annotations
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import tempfile
import os

def enhance_image(path: str, scale: float = 1.0) -> str:
    p = Path(path)
    img = Image.open(p).convert("RGB")
    # basic enhancements
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w*scale), int(h*scale)))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name

def cleanup_temp(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass
