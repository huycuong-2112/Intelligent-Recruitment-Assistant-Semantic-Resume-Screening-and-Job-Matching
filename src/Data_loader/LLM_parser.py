from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    print("Vui lòng cài đặt thư viện groq: pip install groq pydantic python-dotenv")
    exit(1)

# Tự động tìm thư mục gốc Project
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

INPUT_CLEANED_TEXT = project_root / "Data" / "Processed" / "cleaned_text.json"
OUTPUT_JSON = project_root / "Data" / "Processed" / "parsed_resumes.json"

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "gemma2-9b-it"
]


class ResumeData(BaseModel):
    skills: List[str] = Field(
        default_factory=list, 
        description="Mảng các kỹ năng chuyên môn (ví dụ: ['Python', 'SQL', 'Docker', 'CAD'])"
    )
    experience_years: float = Field(
        0.0, 
        description="Tổng số năm kinh nghiệm làm việc. Nếu có mốc năm (VD: 2020-2024), tự trừ lấy số năm; nếu ghi '5+ năm' lấy 5.0; nếu ghi 'hiện tại/present' thì lấy mốc 2026."
    )
    education_degree: Optional[str] = Field(
        None, 
        description="Bằng cấp cao nhất: 'Bachelor', 'Master', 'Engineer', 'Ph.D', 'Associate'. Không có để null."
    )
    education_field: Optional[str] = Field(
        None, 
        description="Chuyên ngành đào tạo (ví dụ: 'Computer Science', 'Mechanical Engineering', 'Finance')"
    )
    job_titles: List[str] = Field(
        default_factory=list, 
        description="Danh sách các chức danh công việc từng đảm nhiệm (ví dụ: ['Data Engineer', 'Embedded Developer'])"
    )


def extract_fallback_regex(text: str) -> dict:
    skills = []
    skill_match = re.search(r'(?:Skills|Kỹ năng|Technical Skills)[\s:]*\n(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
    if skill_match:
        skills = [s.strip() for s in re.split(r'[,•\n]+', skill_match.group(1)) if s.strip()]
        
    edu_degree = None
    text_lower = text.lower()
    if re.search(r'\b(tiến sĩ|ph\.d|doctorate)\b', text_lower):
        edu_degree = "Ph.D"
    elif re.search(r'\b(thạc sĩ|master)\b', text_lower):
        edu_degree = "Master"
    elif re.search(r'\b(kỹ sư|engineer)\b', text_lower):
        edu_degree = "Engineer"
    elif re.search(r'\b(cử nhân|bachelor|bs|ba)\b', text_lower):
        edu_degree = "Bachelor"
        
    edu_field = None
    field_match = re.search(r'(?:chuyên ngành|major)[:\s]+([a-zA-Zà-ỹÀ-Ỹ\s]+)', text, re.IGNORECASE)
    if field_match:
        edu_field = field_match.group(1).strip()

    exp_years = 0.0
    exp_explicit = re.search(r'(\d+(?:\.\d+)?)\+?\s*(?:năm|years?)\s*(?:kinh nghiệm|of experience)', text, re.IGNORECASE)
    if exp_explicit:
        exp_years = float(exp_explicit.group(1))
    else:
        date_ranges = re.findall(r'((?:0[1-9]|1[0-2])?/?(20\d{2}))\s*[-–—tođến]+\s*((?:0[1-9]|1[0-2])?/?(20\d{2})|hiện tại|nay|present|now)', text, re.IGNORECASE)
        total_years = 0.0
        for r in date_ranges:
            start_year = int(r[1])
            end_str = r[2].lower()
            end_year = 2026 if end_str in ['hiện tại', 'nay', 'present', 'now'] else int(r[3])
            if end_year >= start_year:
                total_years += (end_year - start_year)
        if total_years > 0:
            exp_years = float(total_years)

    return {
        "skills": skills,
        "experience_years": exp_years,
        "education_degree": edu_degree,
        "education_field": edu_field,
        "job_titles": []
    }


def parse_resume_llm(text: str, client: Groq) -> dict:
    schema_json = ResumeData.model_json_schema()
    prompt = f"""
    Từ văn bản CV sau, hãy trích xuất thông tin thành JSON hợp lệ theo đúng schema:
    {json.dumps(schema_json, indent=2)}
    
    CV:
    {text[:4000]}
    """
    last_error = None
    for model_name in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a resume parsing engine that strictly outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            last_error = e
            continue
            
    raise last_error


def determine_domain_from_path(relative_path: str) -> str:
    path_lower = relative_path.lower()
    if "it" in path_lower:
        return "IT"
    if "engineer" in path_lower:
        return "Engineering"
    if "economics" in path_lower:
        return "Economics"
    return "Unknown"


def main():
    print("=" * 80)
    print("RESUME ENTITY STRUCTURING (LLM PARSER TẦNG 2)")
    print("=" * 80)

    if not INPUT_CLEANED_TEXT.exists():
        print(f"❌ Không tìm thấy file: {INPUT_CLEANED_TEXT}")
        print("Vui lòng chạy 'python main.py' trước để bóc tách văn bản thô từ tài liệu.")
        return

    with open(INPUT_CLEANED_TEXT, "r", encoding="utf-8") as f:
        cleaned_docs = json.load(f)

    print(f"📂 Đã nạp {len(cleaned_docs)} tài liệu từ {INPUT_CLEANED_TEXT.name}...")

    api_key = os.getenv("GROQ_API_KEY", "")
    client = None
    if api_key and api_key.startswith("gsk_"):
        client = Groq(api_key=api_key)
    else:
        print("⚠️ Không tìm thấy Groq API Key hợp lệ. Chuyển sang chế độ Regex Fallback.")

    parsed_results = []
    success_api = 0
    success_fallback = 0

    for idx, doc in enumerate(cleaned_docs, 1):
        filename = doc.get("filename", "")
        content = doc.get("content", "")
        rel_path = doc.get("relative_path", "")
        domain = determine_domain_from_path(rel_path)

        if not content.strip():
            print(f"   [{idx}/{len(cleaned_docs)}] ⚠️ Bỏ qua (không có nội dung): {filename}")
            continue

        print(f"   [{idx}/{len(cleaned_docs)}] Đang bóc tách [{domain}]: {filename}...")

        parsed_data = None
        if client:
            try:
                parsed_data = parse_resume_llm(content, client)
                success_api += 1
            except Exception as e:
                print(f"      -> [LLM Lỗi, dùng Fallback]: {e}")

        if not parsed_data:
            parsed_data = extract_fallback_regex(content)
            success_fallback += 1

        record = {
            "filename": filename,
            "domain": domain,
            "parsed_data": parsed_data,
            "raw_text_length": len(content),
            "source_status": doc.get("status")
        }
        parsed_results.append(record)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("HOÀN TẤT CẤU TRÚC HÓA THỰC THỂ TẤT CẢ RESUMES")
    print(f"Thành công qua Groq LLM : {success_api}")
    print(f"Thành công qua Regex    : {success_fallback}")
    print(f"Tổng số CV hoàn thành   : {len(parsed_results)}")
    print(f"File kết quả lưu tại    : {OUTPUT_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()