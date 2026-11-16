from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq, RateLimitError
except ImportError:
    Groq = None
    RateLimitError = Exception

try:
    from sentence_transformers import SentenceTransformer, util
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embedder = None

# ---------------------------------------------------------------------------
# Import refactored schema & offline extractor
# ---------------------------------------------------------------------------
try:
    from .jd_schema import DegreeType, StructuredJobDescription
    from .offline.jd_offline_parser import OfflineJDExtractor
except ImportError:
    # Support direct script execution: python src/Data_loader/jd_parser.py
    _current_dir = str(Path(__file__).resolve().parent)
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)
    from jd_schema import DegreeType, StructuredJobDescription
    from offline.jd_offline_parser import OfflineJDExtractor

# ---------------------------------------------------------------------------
# 1. PATHS & CONFIGURATION
# ---------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists() or (project_root / "src").exists():
        break
    project_root = project_root.parent

# Output của Stage 1, được tạo bởi main.py --type jds
INPUT_CLEANED_JDS = project_root / "Data" / "Processed" / "cleaned_jds.json"
OUTPUT_PARSED_JDS = project_root / "Data" / "Processed" / "parsed_jds.json"

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

CANONICAL_FIELDS = [
    "Computer Science & Software Engineering",
    "Robotics & Mechatronics",
    "Electrical & Electronic Engineering",
    "Mechanical Engineering",
    "Data Science & Artificial Intelligence",
    "Business Administration & Economics",
    "Finance & Banking",
    "Marketing & Supply Chain Management",
    "Logistics & Supply Chain Management",
    "Accounting & Auditing"
]


# ---------------------------------------------------------------------------
# 4. ONLINE LLM JD STRUCTURING (GROQ)
# ---------------------------------------------------------------------------
def parse_jd_llm(text: str, client: Any) -> StructuredJobDescription:
    schema_json = StructuredJobDescription.model_json_schema()
    system_prompt = (
        "You are an expert HR Recruitment & Job Description Analysis Engine. "
        "Extract structured requirements from the Job Description text into a valid JSON object strictly matching this schema:\n"
        f"{json.dumps(schema_json, ensure_ascii=False, indent=2)}\n\n"
        "Guidelines:\n"
        "1. Identify the exact job title and hiring company.\n"
        "2. Differentiate between 'required_skills' (mandatory) and 'preferred_skills' (plus points).\n"
        "3. Extract minimum experience years as a clean float (e.g., '1-2 years' -> 1.0, 'fresher' -> 0.0).\n"
        "4. Capture key deliverables and responsibilities without losing technical specifics."
    )

    compact_text = re.sub(r"[ \t]+", " ", text)
    compact_text = re.sub(r"\n{3,}", "\n\n", compact_text).strip()
    user_prompt = f"Job Description Content:\n{compact_text[:10000]}"

    last_error = None
    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            raw = response.choices[0].message.content
            if raw:
                return StructuredJobDescription.model_validate_json(raw)
        except RateLimitError as rle:
            raise rle
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise RuntimeError("All LLM attempts failed.")


# ---------------------------------------------------------------------------
# 5. MAIN PROCESSING CONTROLLER
# ---------------------------------------------------------------------------
def collect_cleaned_jds(input_file: Path) -> List[Tuple[str, str, str]]:
    """Đọc các JD đã được trích xuất text bởi Stage 1."""
    jds: List[Tuple[str, str, str]] = []
    if not input_file.exists():
        return jds

    try:
        records = json.loads(input_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return jds

    if not isinstance(records, list):
        return jds

    for idx, item in enumerate(records, 1):
        if not isinstance(item, dict):
            continue
        text_content = item.get("content", "")
        if isinstance(text_content, str) and text_content.strip():
            jds.append((
                str(item.get("id", f"jd_{idx:03d}")),
                str(item.get("filename", f"jd_{idx:03d}")),
                text_content,
            ))
    return jds

def parse_cleaned_jds(input_path: Path = INPUT_CLEANED_JDS, output_path: Path = OUTPUT_PARSED_JDS, offline: bool = False) -> list[Dict[str, Any]]:
    """Parse Stage-1 cleaned JDs with Groq-first and the separated offline fallback."""
    records = json.loads(Path(input_path).read_text(encoding="utf-8"))
    key = None if offline else os.getenv("GROQ_API_KEY")
    client = Groq(api_key=key) if Groq and key and key.startswith("gsk_") else None
    results = []
    for idx, doc in enumerate(records, 1):
        text = str(doc.get("content", ""))
        if not text.strip():
            continue
        method = "offline_hybrid"; structured = None
        if client:
            try:
                structured = parse_jd_llm(text, client); method = "groq_llm"
            except Exception:
                structured = None
        if structured is None:
            structured = OfflineJDExtractor.parse(text, str(doc.get("filename", f"jd_{idx:03d}")))
        results.append({"id": doc.get("id", f"jd_{idx:03d}"), "filename": doc.get("filename", ""), "extraction_method": method, "source_status": doc.get("status"), "raw_text_length": len(text), "parsed_data": structured.model_dump()})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main():
    print("=" * 80)
    print("JOB DESCRIPTION STRUCTURING (1-TO-1 SCHEMA EXTRACTION)")
    print(f"Input Cleaned    : {INPUT_CLEANED_JDS}")
    print(f"Output File      : {OUTPUT_PARSED_JDS}")
    print("=" * 80)

    raw_jds = collect_cleaned_jds(INPUT_CLEANED_JDS)
    if not raw_jds:
        print(f"⚠️ File '{INPUT_CLEANED_JDS}' chưa có dữ liệu. Hãy chạy main.py --type jds trước.")
        return

    print(f"📂 Tìm thấy {len(raw_jds)} vị trí tuyển dụng (JDs) để bóc tách...\n")

    api_key = os.getenv("GROQ_API_KEY")# Nhap API KEY VAO ()
    groq_client = Groq(api_key=api_key) if (Groq and api_key and api_key.startswith("gsk_")) else None

    if groq_client:
        print("🌐 Chế độ: ONLINE (Groq LLM Engine)")
    else:
        print("🔌 Chế độ: OFFLINE (Hybrid Regex + MiniML Engine)")

    parsed_jds: List[Dict[str, Any]] = []
    online_count = 0
    offline_count = 0

    for idx, (jd_id, default_title, raw_text) in enumerate(raw_jds, 1):
        if not raw_text.strip():
            continue

        print(f"[{idx}/{len(raw_jds)}] Đang cấu trúc hóa JD: {default_title}...")
        structured_jd: Optional[StructuredJobDescription] = None
        method = "offline_hybrid"

        if groq_client:
            try:
                structured_jd = parse_jd_llm(raw_text, groq_client)
                method = "groq_llm"
                online_count += 1
            except RateLimitError:
                print("   └─ ⚠️ Rate limit 429. Chuyển sang Offline Engine...")
            except Exception as e:
                print(f"   └─ ⚠️ LLM Error ({type(e).__name__}). Dùng Offline Fallback...")

        if structured_jd is None:
            structured_jd = OfflineJDExtractor.parse(raw_text, default_title)
            method = "offline_hybrid"
            offline_count += 1

        print(
            f"   └─ Vị trí: {structured_jd.job_title} | Yêu cầu: {structured_jd.min_experience_years} năm | "
            f"Bằng cấp: {structured_jd.required_degree} | Skills: {len(structured_jd.required_skills)}"
        )

        parsed_jds.append({
            "id": jd_id,
            "extraction_method": method,
            "raw_text_length": len(raw_text),
            "parsed_data": structured_jd.model_dump()
        })

    OUTPUT_PARSED_JDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PARSED_JDS, "w", encoding="utf-8") as f:
        json.dump(parsed_jds, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("HOÀN TẤT CẤU TRÚC HÓA JDs")
    print(f"Tổng số JDs lưu       : {len(parsed_jds)}")
    print(f"Xử lý qua Online LLM  : {online_count}")
    print(f"Xử lý qua Offline     : {offline_count}")
    print(f"File kết quả          : {OUTPUT_PARSED_JDS}")
    print("=" * 80)


if __name__ == "__main__":
    main()
