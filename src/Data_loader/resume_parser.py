from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. OPTIONAL DEPENDENCIES FOR LLM
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Import refactored schema & offline parser
# ---------------------------------------------------------------------------
try:
    from .resume_schema import StructuredResume
    from .offline.resume_offline_parser import OfflineResumeParser
except ImportError:
    # Support direct script execution: python src/Data_loader/resume_parser.py
    _current_dir = str(Path(__file__).resolve().parent)
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)
    from resume_schema import StructuredResume
    from offline.resume_offline_parser import OfflineResumeParser

# ---------------------------------------------------------------------------
# 2. PATHS & CONFIGURATION
# ---------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists() or (project_root / "src").exists():
        break
    project_root = project_root.parent

INPUT_CLEANED_RESUMES = project_root / "Data" / "Processed" / "cleaned_resumes.json"
OUTPUT_PARSED_JSON = project_root / "Data" / "Processed" / "parsed_resumes.json"

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]


# ---------------------------------------------------------------------------
# 5. ONLINE LLM STRUCTURING ENGINE (GROQ)
# ---------------------------------------------------------------------------
def parse_resume_llm(text: str, client: Any) -> StructuredResume:
    schema_json = StructuredResume.model_json_schema()
    system_prompt = (
        "You are an expert AI resume parsing and candidate evaluation engine. "
        "Extract all structured entities from the CV text into a valid JSON object strictly matching this schema:\n"
        f"{json.dumps(schema_json, ensure_ascii=False, indent=2)}\n\n"
        "Strict Extraction Rules:\n"
        "1. WORK EXPERIENCE: Capture ALL roles without omission. If month/year is not explicitly stated next to a role, scan the surrounding text for dates or estimate duration.\n"
        "2. SKILLS HARVESTING: If there is no dedicated 'Skills' section, harvest all programming languages, tools, frameworks, and methodologies mentioned in summary, projects, and work experience.\n"
        "3. ENTITY CLEANING: Sanitize institution and company names to remove accidental glued job titles, OCR noise, or watermarks (e.g., 'Trường Cao đẳng Công Thương' instead of 'Trường Cao đẳng Công Thức tập sinh').\n"
        "4. PROJECTS & METRICS: Extract all quantifiable metrics (%, FPS, latency, scale, revenue, generations) into 'impact_metrics'.\n"
        "5. Current reference year is 2026."
    )

    # Nén khoảng trắng & giới hạn 12,000 ký tự (đủ cho CV 3-4 trang, an toàn token)
    compact_text = re.sub(r"[ \t]+", " ", text)
    compact_text = re.sub(r"\n{3,}", "\n\n", compact_text).strip()
    user_prompt = f"CV Content:\n{compact_text[:12000]}"

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
            raw_content = response.choices[0].message.content
            if raw_content:
                return StructuredResume.model_validate_json(raw_content)
        except RateLimitError as rle:
            raise rle  # Bắn lỗi 429 ra ngoài để kích hoạt chuyển sang Offline Parser
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise RuntimeError("All LLM model candidates failed.")


def determine_domain(relative_path: str) -> str:
    p = relative_path.lower()
    if "it" in p:
        return "IT"
    if "engineer" in p:
        return "Engineering"
    if "economics" in p:
        return "Economics"
    return "General"


# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE CONTROLLER
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STAGE 2: ENTITY STRUCTURING & PROJECT ASSESSMENT")
    print(f"Project Root    : {project_root}")
    print(f"Input Cleaned   : {INPUT_CLEANED_RESUMES}")
    print(f"Output Parsed   : {OUTPUT_PARSED_JSON}")
    print("=" * 80)

    if not INPUT_CLEANED_RESUMES.exists():
        print(f"❌ Error: {INPUT_CLEANED_RESUMES} not found. Please run main.py (Stage 1) first.")
        return

    with open(INPUT_CLEANED_RESUMES, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"📂 Loaded {len(docs)} documents from Stage 1...\n")

    api_key = os.getenv("GROQ_API_KEY")
    # api_key = os.getenv("GROQ_API_KEY")
    groq_client = Groq(api_key=api_key) if (Groq and api_key and api_key.startswith("gsk_")) else None

    if groq_client:
        print("🌐 Mode: ONLINE (Groq LLM Engine Active)")
    else:
        print("🔌 Mode: OFFLINE (Hybrid Regex + MiniML Engine Active)")

    results: List[Dict[str, Any]] = []
    online_count = 0
    offline_count = 0

    for idx, doc in enumerate(docs, 1):
        cv_id = doc.get("id", f"cv_{idx:03d}")
        filename = doc.get("filename", "")
        content = doc.get("content", "")
        rel_path = doc.get("relative_path", "")
        domain = determine_domain(rel_path)

        if not content.strip():
            print(f"[{idx}/{len(docs)}] ⚠️ Skipping empty document: {filename}")
            continue

        print(f"[{idx}/{len(docs)}] Structuring [{domain}]: {filename}...")

        structured_data: Optional[StructuredResume] = None
        method_used = "offline_hybrid"

        # 1. Thử phân tích qua Groq LLM nếu có API key
        if groq_client:
            try:
                structured_data = parse_resume_llm(content, groq_client)
                method_used = "groq_llm"
                online_count += 1
            except RateLimitError:
                print("   └─ ⚠️ Rate limit (429) hit. Gracefully falling back to Offline Engine...")
            except Exception as exc:
                print(f"   └─ ⚠️ LLM Error ({type(exc).__name__}). Using Offline Fallback...")

        # 2. Kích hoạt Offline Fallback nếu LLM không khả dụng hoặc bị giới hạn rate limit
        if structured_data is None:
            structured_data = OfflineResumeParser.parse(content)
            method_used = "offline_hybrid"
            offline_count += 1

        print(
            f"   └─ Method: {method_used} | Degree: {structured_data.education_degree} | "
            f"Exp: {structured_data.experience_years} yrs | Roles: {len(structured_data.work_experience)} | "
            f"Projects: {len(structured_data.projects)} | Skills: {len(structured_data.skills)}"
        )

        results.append({
            "id": cv_id,
            "filename": filename,
            "domain": domain,
            "extraction_method": method_used,
            "source_status": doc.get("status"),
            "parsed_data": structured_data.model_dump(),
        })

    OUTPUT_PARSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PARSED_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 2 STRUCTURING COMPLETE")
    print(f"Total Processed       : {len(results)}")
    print(f"Parsed via Online LLM : {online_count}")
    print(f"Parsed via Offline    : {offline_count}")
    print(f"Output File           : {OUTPUT_PARSED_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
