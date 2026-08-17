import os
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

try:
    from groq import Groq
except ImportError:
    print("Vui lòng cài đặt thư viện groq: pip install groq pydantic")
    exit(1)

# Tự động tìm thư mục gốc Project chứa folder "Data"
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

# Đường dẫn trỏ thẳng vào Data/Raw/Resumes chứa 4 thư mục ngành
RAW_RESUMES_DIR = project_root / "Data" / "Raw" / "Resumes"
OUTPUT_JSON = project_root / "Data" / "Processed" / "parsed_resumes.json"

DOMAINS = ["IT", "Engineer", "Economics", "Healthcare"]

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
    {text}
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a resume parsing engine that strictly outputs valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)

def main():
    print("=" * 80)
    print("RESUME PARSER THEO ĐÚNG CẤU TRÚC FOLDER RESUMES")
    print("=" * 80)

    api_key = os.getenv("GROQ_API_KEY")
    
    client = None
    if api_key and api_key.startswith("gsk_"):
        client = Groq(api_key=api_key)
    else:
        print("⚠️ Không tìm thấy Groq API Key hợp lệ trong biến môi trường (GROQ_API_KEY). Chuyển sang chế độ Regex Fallback.")

    parsed_results = []
    success_api = 0
    success_fallback = 0

    for domain in DOMAINS:
        domain_folder = RAW_RESUMES_DIR / domain
        if not domain_folder.exists():
            print(f"⚠️ Thư mục chưa tồn tại: {domain_folder}")
            continue

        text_files = list(domain_folder.glob("*.txt"))
        print(f"\n📂 Đang xử lý Domain [{domain}] - {len(text_files)} files...")

        for idx, file_path in enumerate(text_files, 1):
            with open(file_path, "r", encoding="utf-8") as f:
                cv_text = f.read()

            print(f"   [{idx}/{len(text_files)}] File: {file_path.name}...")

            parsed_data = None
            if client:
                try:
                    parsed_data = parse_resume_llm(cv_text, client)
                    success_api += 1
                except Exception as e:
                    print(f"      -> [API Lỗi]: {e}")

            if not parsed_data:
                parsed_data = extract_fallback_regex(cv_text)
                success_fallback += 1

            record = {
                "filename": file_path.name,
                "domain": domain,
                "parsed_data": parsed_data,
                "raw_text_length": len(cv_text)
            }
            parsed_results.append(record)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("HOÀN TẤT XỬ LÝ TẤT CẢ RESUMES")
    print(f"Thành công qua Groq API : {success_api}")
    print(f"Thành công qua Regex    : {success_fallback}")
    print(f"Tổng số CV đã xử lý     : {len(parsed_results)}")
    print(f"File lưu tại            : {OUTPUT_JSON}")
    print("=" * 80)

if __name__ == "__main__":
    main()