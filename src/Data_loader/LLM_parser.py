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

# Đọc PDF
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Đọc DOCX
try:
    import docx
except ImportError:
    docx = None

# Tích hợp bộ bóc tách ảnh & OCR Fallback
try:
    from document_parser import get_document_parser
except ImportError:
    try:
        from src.Data_loader.document_parser import get_document_parser
    except ImportError:
        get_document_parser = None

# Tự động tìm thư mục gốc Project
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

RAW_RESUMES_DIR = project_root / "Data" / "Raw" / "Resumes"
OUTPUT_JSON = project_root / "Data" / "Processed" / "parsed_resumes.json"

# 3 Domains trọng tâm
DOMAINS = ["IT", "Engineering", "Economics"]

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


def extract_text_from_file(file_path: Path) -> str:
    """Bóc tách nội dung văn bản hỗ trợ PDF, DOCX, TXT và các file ẢNH (PNG, JPG, JPEG)."""
    ext = file_path.suffix.lower()
    text = ""
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif ext == ".pdf":
            if PdfReader is not None:
                reader = PdfReader(str(file_path), strict=False)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            # Nếu PDF dạng scan không có text layer, gọi OCR parser
            if not text.strip() and get_document_parser is not None:
                parser = get_document_parser()
                text, _ = parser.parse(str(file_path))

        elif ext in [".docx", ".doc"]:
            if docx is not None:
                doc = docx.Document(str(file_path))
                text = "\n".join([p.text for p in doc.paragraphs if p.text])

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            if get_document_parser is not None:
                parser = get_document_parser()
                text, _ = parser.parse(str(file_path))

    except Exception as e:
        print(f"      [Lỗi đọc file {file_path.name}]: {e}")

    return text.strip()


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


def find_domain_folder(base_dir: Path, domain_name: str) -> Optional[Path]:
    if (base_dir / domain_name).exists():
        return base_dir / domain_name
    for folder in base_dir.iterdir():
        if folder.is_dir() and domain_name.lower() in folder.name.lower():
            return folder
    return None


def main():
    print("=" * 80)
    print("RESUME PARSER THEO 3 DOMAINS (IT, Engineering, Economics)")
    print("Hỗ trợ định dạng: PDF, DOCX, TXT, PNG, JPG, JPEG")
    print("=" * 80)

    # Đọc API key từ biến môi trường hệ thống hoặc file .env
    api_key = os.getenv("GROQ_API_KEY", "")
    
    # Nếu chưa có biến môi trường, bạn có thể gán tạm thời khi chạy local:
    # api_key = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY"

    client = None
    if api_key and api_key.startswith("gsk_"):
        client = Groq(api_key=api_key)
    else:
        print("⚠️ Không tìm thấy Groq API Key hợp lệ. Chuyển sang chế độ Regex Fallback.")

    parsed_results = []
    success_api = 0
    success_fallback = 0

    SUPPORTED_EXTS = [".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"]

    for domain in DOMAINS:
        domain_folder = find_domain_folder(RAW_RESUMES_DIR, domain)
        if not domain_folder or not domain_folder.exists():
            print(f"⚠️ Thư mục chưa tồn tại cho Domain [{domain}] tại: {RAW_RESUMES_DIR}")
            continue

        resume_files = [f for f in domain_folder.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
        print(f"\n📂 Đang xử lý Domain [{domain}] (Folder: {domain_folder.name}) - {len(resume_files)} files...")

        for idx, file_path in enumerate(resume_files, 1):
            cv_text = extract_text_from_file(file_path)
            if not cv_text:
                print(f"   [{idx}/{len(resume_files)}] ⚠️ Không đọc được text: {file_path.name}")
                continue

            print(f"   [{idx}/{len(resume_files)}] Đang parse: {file_path.name}...")
            parsed_data = None
            if client:
                try:
                    parsed_data = parse_resume_llm(cv_text, client)
                    success_api += 1
                except Exception as e:
                    print(f"      -> [API Lỗi, dùng Fallback]: {e}")

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