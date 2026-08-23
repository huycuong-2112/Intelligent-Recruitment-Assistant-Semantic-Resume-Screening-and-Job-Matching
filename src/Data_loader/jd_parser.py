from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

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
# 1. PATHS & CONFIGURATION
# ---------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists() or (project_root / "src").exists():
        break
    project_root = project_root.parent

# Thư mục chứa JDs cào về (chấp nhận file .json, .txt hoặc markdown)
INPUT_RAW_JDS = project_root / "Data" / "Raw" / "JDs"
OUTPUT_PARSED_JDS = project_root / "Data" / "Processed" / "parsed_jds.json"

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

DegreeType = Literal["High School", "Associate", "Bachelor", "Engineer", "Master", "Ph.D", "Any", "Other"]

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
# 2. 1-TO-1 JD PYDANTIC SCHEMA
# ---------------------------------------------------------------------------
class StructuredJobDescription(BaseModel):
    job_title: str = Field(..., description="Chức danh công việc cần tuyển dụng")
    company_name: Optional[str] = Field(None, description="Tên công ty / doanh nghiệp tuyển dụng")
    job_overview: Optional[str] = Field(None, description="Mô tả tóm tắt vai trò và sứ mệnh của vị trí")
    min_experience_years: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="Số năm kinh nghiệm tối thiểu yêu cầu (VD: ghi '2+ năm' lấy 2.0; không yêu cầu lấy 0.0)"
    )
    required_degree: Optional[DegreeType] = Field(
        default=None,
        description="Bằng cấp tối thiểu: 'Associate', 'Bachelor', 'Engineer', 'Master', 'Ph.D', 'Any'"
    )
    preferred_fields: List[str] = Field(
        default_factory=list,
        description="Danh sách các chuyên ngành đào tạo ưu tiên (VD: ['Computer Science', 'Mechatronics'])"
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Kỹ năng kỹ thuật và nghiệp vụ BẮT BUỘC (Must-have)"
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Kỹ năng ưu tiên / điểm cộng (Nice-to-have)"
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Danh sách các đầu việc và trách nhiệm chính của vị trí"
    )
    key_deliverables: List[str] = Field(
        default_factory=list,
        description="Các dự án hoặc bài toán cốt lõi cần giải quyết"
    )
    required_certifications: List[str] = Field(
        default_factory=list,
        description="Chứng chỉ bắt buộc hoặc ưu tiên (TOEIC, IELTS, CFA, AWS, PMP...)"
    )

    @field_validator("required_skills", "preferred_skills", "preferred_fields", "required_certifications", mode="before")
    @classmethod
    def deduplicate_list(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if isinstance(item, str):
                s = item.strip().strip("•-* \t\n")
                if s and s not in cleaned:
                    cleaned.append(s)
        return cleaned

    @field_validator("min_experience_years", mode="before")
    @classmethod
    def clean_exp_years(cls, value: Any) -> float:
        try:
            return max(0.0, round(float(value), 1))
        except (ValueError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# 3. OFFLINE HEURISTIC JD EXTRACTOR (REGEX + MiniLM)
# ---------------------------------------------------------------------------
class OfflineJDExtractor:
    @staticmethod
    def extract_min_experience(text: str) -> float:
        match = re.search(
            r'(\d+(?:\.\d+)?)\+?\s*(?:năm|years?)\s*(?:kinh nghiệm|of experience|experience required)',
            text,
            re.IGNORECASE
        )
        if match:
            return float(match.group(1))
        if re.search(r'\b(fresher|intern|không yêu cầu kinh nghiệm|no experience required)\b', text, re.IGNORECASE):
            return 0.0
        return 0.0

    @staticmethod
    def extract_degree(text: str) -> Optional[DegreeType]:
        text_lower = text.lower()
        if re.search(r'\b(tiến sĩ|ph\.d|doctorate)\b', text_lower):
            return "Ph.D"
        if re.search(r'\b(thạc sĩ|master)\b', text_lower):
            return "Master"
        if re.search(r'\b(kỹ sư|engineer)\b', text_lower):
            return "Engineer"
        if re.search(r'\b(cử nhân|bachelor|đại học|university degree)\b', text_lower):
            return "Bachelor"
        if re.search(r'\b(cao đẳng|associate)\b', text_lower):
            return "Associate"
        return "Any"

    @classmethod
    def extract_skills_and_responsibilities(cls, text: str) -> Tuple[List[str], List[str], List[str]]:
        req_skills: List[str] = []
        resp: List[str] = []
        certs: List[str] = []

        for line in text.splitlines():
            line_clean = line.strip().strip("•-* ")
            if not line_clean:
                continue

            # Bóc tách chứng chỉ
            if re.search(r'\b(TOEIC|IELTS|CFA|AWS|Azure|GCP|PMP|JLPT|HSK)\b', line_clean, re.IGNORECASE):
                certs.append(line_clean)

            # Phân loại bullet trách nhiệm
            if len(line_clean) > 30 and re.search(r'\b(phát triển|thiết kế|xây dựng|quản lý|triển khai|chịu trách nhiệm|develop|design|manage|maintain|implement)\b', line_clean, re.IGNORECASE):
                resp.append(line_clean)
            elif 2 < len(line_clean) <= 30 and not line_clean.startswith("#"):
                tokens = re.split(r'[,/|]+', line_clean)
                for t in tokens:
                    t_str = t.strip()
                    if 1 < len(t_str) <= 25 and t_str not in req_skills:
                        req_skills.append(t_str)

        return req_skills[:20], resp[:10], certs[:5]

    @classmethod
    def parse(cls, text: str, fallback_title: str) -> StructuredJobDescription:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        title = lines[0] if lines and len(lines[0]) < 60 else fallback_title
        title = re.sub(r'^[#*_\-\s]+', '', title)

        req_skills, responsibilities, certs = cls.extract_skills_and_responsibilities(text)
        exp_years = cls.extract_min_experience(text)
        degree = cls.extract_degree(text)

        return StructuredJobDescription(
            job_title=title,
            company_name=None,
            job_overview=text[:350].strip(),
            min_experience_years=exp_years,
            required_degree=degree,
            preferred_fields=[],
            required_skills=req_skills,
            preferred_skills=[],
            responsibilities=responsibilities,
            key_deliverables=[],
            required_certifications=certs
        )


# ---------------------------------------------------------------------------
# 4. ONLINE LLM JD STRUCTURING (GROQ)
# ---------------------------------------------------------------------------
def parse_jd_llm(text: str, client: Groq) -> StructuredJobDescription:
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
def collect_raw_jds(input_dir: Path) -> List[Tuple[str, str, str]]:
    """Đọc toàn bộ file JDs cào về (hỗ trợ .json, .txt, .md)."""
    jds: List[Tuple[str, str, str]] = []
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        return jds

    for idx, f in enumerate(sorted(input_dir.glob("*")), 1):
        if f.is_file():
            jd_id = f"jd_{idx:03d}"
            try:
                content = f.read_text(encoding="utf-8")
                # Nếu là JSON cào từ web (TopCV, LinkedIn, v.v.)
                if f.suffix.lower() == ".json":
                    data = json.loads(content)
                    if isinstance(data, list):
                        for j_idx, item in enumerate(data, 1):
                            text_content = item.get("content") or item.get("description") or json.dumps(item, ensure_ascii=False)
                            jds.append((f"jd_{idx:03d}_{j_idx:02d}", item.get("title", f.stem), text_content))
                        continue
                    elif isinstance(data, dict):
                        text_content = data.get("content") or data.get("description") or json.dumps(data, ensure_ascii=False)
                        jds.append((jd_id, data.get("title", f.stem), text_content))
                        continue
                jds.append((jd_id, f.stem, content))
            except Exception:
                continue
    return jds


def main():
    print("=" * 80)
    print("JOB DESCRIPTION STRUCTURING (1-TO-1 SCHEMA EXTRACTION)")
    print(f"Input Directory  : {INPUT_RAW_JDS}")
    print(f"Output File      : {OUTPUT_PARSED_JDS}")
    print("=" * 80)

    raw_jds = collect_raw_jds(INPUT_RAW_JDS)
    if not raw_jds:
        print(f"⚠️ Thư mục '{INPUT_RAW_JDS}' chưa có file JD nào. Đặt các file .txt/.json vào thư mục này rồi chạy lại.")
        return

    print(f"📂 Tìm thấy {len(raw_jds)} vị trí tuyển dụng (JDs) để bóc tách...\n")

    api_key = os.getenv("GROQ_API_KEY")
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