from __future__ import annotations
import re

def fix_glued_text(text: str) -> str:
    """Restores missing spaces between words, numbers, and bullet points."""
    if not text:
        return ""

    t = text
    # Lowercase attached to Uppercase
    t = re.sub(r"([a-zà-ỹ])([A-ZÀ-Ỹ])", r"\1 \2", t)
    # Letter attached to Number
    t = re.sub(r"([a-zA-Zà-ỹÀ-Ỹ])(\d)", r"\1 \2", t)
    # Number attached to Letter
    t = re.sub(r"(\d)([a-zA-Zà-ỹÀ-Ỹ])", r"\1 \2", t)
    # Separate bullets and dashes
    t = re.sub(r"([a-zA-Zà-ỹÀ-Ỹ0-9])([•|\-/—])", r"\1 \2", t)
    t = re.sub(r"([•|\-/—])([a-zA-Zà-ỹÀ-Ỹ0-9])", r"\1 \2", t)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    return "\n".join(lines).strip()