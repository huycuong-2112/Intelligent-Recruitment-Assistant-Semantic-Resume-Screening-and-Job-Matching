from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# 1. OPTIONAL DEPENDENCIES FOR OFFLINE MiniML & LLM
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

try:
    from sentence_transformers import SentenceTransformer, util
    # Mô hình MiniLM siêu nhẹ (~80MB), tối ưu cho CPU
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embedder = None

# ---------------------------------------------------------------------------
# 2. PATHS & CONFIGURATION
# ---------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists() or (project_root / "src").exists():
        break
    project_root = project_root.parent

INPUT_CLEANED_TEXT = project_root / "Data" / "Processed" / "cleaned_text.json"
OUTPUT_PARSED_JSON = project_root / "Data" / "Processed" / "parsed_resumes.json"

GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

DegreeType = Literal["High School", "Associate", "Bachelor", "Engineer", "Master", "Ph.D", "Other"]

CANONICAL_FIELDS = [
    "Computer Science & Software Engineering",
    "Robotics & Mechatronics",
    "Electrical & Electronic Engineering",
    "Mechanical Engineering",
    "Data Science & Artificial Intelligence",
    "Business Administration & Economics",
    "Finance & Banking",
    "Marketing & Supply Chain Management",
    "Accounting & Auditing"
]

IMPACT_PATTERN = re.compile(
    r'(\b\d+(?:\.\d+)?%\b|\b\d+x\b|\b\$\d+[\d,]*\+?\b|\b\d+\s*(?:ms|fps|kb|mb|gb|ghz|users|req/s|stars?|clients?|students?)\b|\bgiảm \d+|\btăng \d+|\bđạt \d+)',
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r'(https?://(?:www\.)?(?:github\.com|gitlab\.com|linkedin\.com|huggingface\.co|[a-zA-Z0-9-]+\.(?:io|dev|app|com))/[^\s\)\],]+)',
    re.IGNORECASE
)

DATE_SPAN_REGEX = re.compile(
    r'((?:(?:0?[1-9]|1[0-2]|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Tháng\s*\d{1,2})[/\.\s,-]*)?(?:20\d{2}|19\d{2}))\s*(?:[-–—tođến\s]+)\s*((?:(?:0?[1-9]|1[0-2]|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Tháng\s*\d{1,2})[/\.\s,-]*)?(?:20\d{2}|19\d{2})|present|now|nay|hiện tại)',
    re.IGNORECASE
)

SECTION_SPLIT_REGEX = re.compile(
    r'(?i)(?:^|\n)(?:#+\s*|\*\*\s*|\b)(EDUCATION|HỌC VẤN|WORK EXPERIENCE|EXPERIENCE|KINH NGHIỆM|PROJECTS|FEATURED PROJECTS|DỰ ÁN|TECHNICAL SKILLS|SKILLS|KỸ NĂNG|CERTIFICATIONS|CHỨNG CHỈ|SUMMARY|CAREER OBJECTIVE|CAREER OBJECTIVES|MỤC TIÊU NGHỀ NGHIỆP|TÓM TẮT)(?:[:\s*\-_=]*)(?:\n|$)'
)


# ---------------------------------------------------------------------------
# 3. PYDANTIC SCHEMA DEFINITIONS
# ---------------------------------------------------------------------------
class EducationItem(BaseModel):
    institution: str = Field(..., description="Tên trường đại học / cao đẳng / viện đào tạo")
    degree: Optional[DegreeType] = Field(None, description="Bằng cấp cao nhất đạt được tại trường")
    field_of_study: Optional[str] = Field(None, description="Chuyên ngành đào tạo")
    start_year: Optional[int] = Field(None, description="Năm bắt đầu học")
    end_year: Optional[int] = Field(None, description="Năm tốt nghiệp (hoặc dự kiến tốt nghiệp)")
    gpa_or_honors: Optional[str] = Field(None, description="Điểm GPA hoặc học bổng/danh hiệu đạt được")


class ExperienceItem(BaseModel):
    company: str = Field(..., description="Tên công ty / tổ chức / doanh nghiệp làm việc")
    role: str = Field(..., description="Chức danh / vị trí công việc")
    start_date: Optional[str] = Field(None, description="Thời gian bắt đầu (VD: '10/2023', '2022')")
    end_date: Optional[str] = Field(None, description="Thời gian kết thúc (VD: '05/2024', 'Present')")
    responsibilities_and_impact: List[str] = Field(
        default_factory=list,
        description="Danh sách các đầu việc, trách nhiệm và thành tựu đã thực hiện"
    )


class ProjectItem(BaseModel):
    name: str = Field(..., description="Tên dự án hoặc đề tài kỹ thuật")
    role: Optional[str] = Field(None, description="Vai trò trong dự án (VD: Leader, Developer, Researcher)")
    technologies: List[str] = Field(default_factory=list, description="Công cụ, ngôn ngữ, framework sử dụng")
    description: str = Field(..., description="Mô tả bài toán kỹ thuật, giải pháp và tính năng")
    impact_metrics: List[str] = Field(
        default_factory=list,
        description="Các chỉ số định lượng đo lường hiệu năng (VD: 'Tăng 30% doanh thu', '45 FPS')"
    )
    links: List[str] = Field(default_factory=list, description="Link demo, source code GitHub hoặc Paper")
    heuristic_score: Optional[float] = Field(
        None,
        description="Điểm đánh giá sơ bộ độ phức tạp và tính định lượng của dự án (0.0 - 1.0)"
    )


class StructuredResume(BaseModel):
    summary: Optional[str] = Field(None, description="Tóm tắt hồ sơ năng lực / Mục tiêu nghề nghiệp")
    education_degree: Optional[DegreeType] = Field(None, description="Bằng cấp cao nhất của ứng viên")
    education_field: Optional[str] = Field(None, description="Chuyên ngành đào tạo chính")
    education_history: List[EducationItem] = Field(default_factory=list, description="Lịch sử học vấn")
    experience_years: float = Field(
        0.0,
        ge=0.0,
        le=50.0,
        description="Tổng số năm kinh nghiệm làm việc tích lũy (làm tròn 1 chữ số thập phân). Mốc hiện tại tính đến 2026."
    )
    job_titles: List[str] = Field(default_factory=list, description="Các chức danh công việc đã đảm nhiệm")
    work_experience: List[ExperienceItem] = Field(
        default_factory=list,
        description="Danh sách toàn bộ các công ty và vị trí công việc từng làm"
    )
    projects: List[ProjectItem] = Field(
        default_factory=list,
        description="Danh sách các dự án tiêu biểu, đồ án hoặc nghiên cứu"
    )
    skills: List[str] = Field(default_factory=list, description="Toàn bộ kỹ năng chuyên môn kỹ thuật và mềm")
    certifications: List[str] = Field(default_factory=list, description="Chứng chỉ chuyên môn hoặc giấy phép hành nghề")

    @field_validator("skills", "job_titles", "certifications", mode="before")
    @classmethod
    def deduplicate_string_list(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if isinstance(item, str):
                s = item.strip().strip("•-* \t")
                if s and s not in cleaned:
                    cleaned.append(s)
        return cleaned

    @field_validator("experience_years", mode="before")
    @classmethod
    def clean_exp_years(cls, value: Any) -> float:
        try:
            return max(0.0, round(float(value), 1))
        except (ValueError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# 4. HEURISTIC & OFFLINE EXTRACTION ENGINE (REGEX + MiniML)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4. ENHANCED OFFLINE RESUME PARSER (REGEX + MiniML HEURISTICS)
# ---------------------------------------------------------------------------
class OfflineResumeParser:

    @staticmethod
    def calculate_heuristic_score(proj_desc: str, techs: List[str], metrics: List[str]) -> float:
        score = 0.0
        score += min(0.35, len(techs) * 0.07)
        score += min(0.35, len(metrics) * 0.15)
        words = len(proj_desc.split())
        if words >= 50:
            score += 0.30
        elif words >= 20:
            score += (words - 20) / (50 - 20) * 0.30
        return round(min(1.0, score), 2)

    @classmethod
    def isolate_sections(cls, text: str) -> Dict[str, str]:
        """Phân đoạn văn bản toàn vẹn thành các section riêng biệt để tránh rò rỉ dữ liệu chéo."""
        sections = {
            "summary": "",
            "education": "",
            "experience": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "other": ""
        }

        matches = []
        for m in SECTION_SPLIT_REGEX.finditer(text):
            tag = m.group(1).upper()
            sec_type = "other"
            if any(k in tag for k in ["EDUCATION", "HỌC VẤN"]):
                sec_type = "education"
            elif any(k in tag for k in ["WORK EXPERIENCE", "EXPERIENCE", "KINH NGHIỆM"]):
                sec_type = "experience"
            elif any(k in tag for k in ["PROJECT", "DỰ ÁN"]):
                sec_type = "projects"
            elif any(k in tag for k in ["SKILL", "KỸ NĂNG"]):
                sec_type = "skills"
            elif any(k in tag for k in ["CERTIFICATION", "CHỨNG CHỈ"]):
                sec_type = "certifications"
            elif any(k in tag for k in ["SUMMARY", "OBJECTIVE", "TÓM TẮT", "MỤC TIÊU"]):
                sec_type = "summary"

            matches.append((m.start(), m.end(), sec_type))

        if not matches:
            sections["other"] = text
            return sections

        matches.sort(key=lambda x: x[0])
        if matches[0][0] > 0:
            sections["summary"] = text[:matches[0][0]].strip()

        for i, (start, end, sec_type) in enumerate(matches):
            next_start = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            content = text[end:next_start].strip()
            if content:
                if sections[sec_type]:
                    sections[sec_type] += "\n\n" + content
                else:
                    sections[sec_type] = content

        return sections

    @classmethod
    def extract_work_experience(cls, exp_text: str) -> Tuple[float, List[str], List[ExperienceItem]]:
        """Bóc tách chính xác từng công ty, chức danh, timeline và tính số năm kinh nghiệm."""
        if not exp_text.strip():
            return 0.0, [], []

        roles_found: List[ExperienceItem] = []
        job_titles: List[str] = []
        total_exp_years = 0.0

        # Tách các block kinh nghiệm dựa trên mốc thời gian
        lines = [l.strip() for l in exp_text.splitlines() if l.strip()]
        current_company = ""
        current_role = ""
        current_start = None
        current_end = None
        current_bullets: List[str] = []

        def save_current_role():
            nonlocal current_company, current_role, current_start, current_end, current_bullets
            if current_company or current_role or current_bullets:
                comp = current_company or "Unknown Organization"
                r = current_role or "Contributor"
                roles_found.append(ExperienceItem(
                    company=comp,
                    role=r,
                    start_date=current_start,
                    end_date=current_end,
                    responsibilities_and_impact=list(current_bullets)
                ))
                if r not in job_titles and r != "Contributor":
                    job_titles.append(r)
            current_company = ""
            current_role = ""
            current_start = None
            current_end = None
            current_bullets = []

        for line in lines:
            # Loại bỏ ký tự markdown thừa
            cleaned_line = re.sub(r'^[#*_\-\s]+', '', line).strip()
            date_match = DATE_SPAN_REGEX.search(cleaned_line)

            if date_match:
                save_current_role()
                current_start = date_match.group(1).strip()
                current_end = date_match.group(2).strip()

                # Đoạn text còn lại trên cùng dòng date thường là Company hoặc Title
                rem_text = DATE_SPAN_REGEX.sub('', cleaned_line).strip('|-–— :')
                if rem_text:
                    if any(kw in rem_text.lower() for kw in ["intern", "engineer", "specialist", "agent", "lead", "staff", "nhân viên", "trưởng nhóm", "kỹ sư"]):
                        current_role = rem_text
                    else:
                        current_company = rem_text
            elif cleaned_line.startswith(('-', '•', '*', 'o ', '+')) or (current_start and not current_company):
                bullet_clean = re.sub(r'^[-•*o+\s]+', '', cleaned_line).strip()
                if bullet_clean:
                    current_bullets.append(bullet_clean)
            elif not current_company and not current_bullets:
                # Dòng trước date thường là Title hoặc Company
                if any(kw in cleaned_line.lower() for kw in ["intern", "engineer", "specialist", "agent", "lead", "staff", "nhân viên", "kỹ sư", "manager"]):
                    current_role = cleaned_line
                else:
                    current_company = cleaned_line

        save_current_role()

        # Tính tổng số năm kinh nghiệm chỉ trong phạm vi Work Experience
        date_ranges = DATE_SPAN_REGEX.findall(exp_text)
        for r in date_ranges:
            start_m = re.search(r'\b(20\d{2}|19\d{2})\b', r[0])
            end_m = re.search(r'\b(20\d{2}|19\d{2})\b', r[1])
            start_yr = int(start_m.group(1)) if start_m else None
            end_str = r[1].lower()
            end_yr = 2026 if any(k in end_str for k in ['present', 'now', 'nay', 'hiện tại']) else (int(end_m.group(1)) if end_m else None)

            if start_yr and end_yr and end_yr >= start_yr:
                total_exp_years += (end_yr - start_yr)

        return min(40.0, float(round(total_exp_years, 1))), job_titles, roles_found

    @classmethod
    def extract_projects(cls, proj_text: str, detected_skills: List[str]) -> List[ProjectItem]:
        """
        Tách rời từng dự án riêng biệt, tự động ghép nối tiêu đề bị ngắt dòng,
        loại bỏ các block metadata/ngày tháng giả mạo và tính điểm heuristic độc lập.
        """
        if not proj_text.strip():
            return []

        # 1. Tách các block ứng viên theo Markdown Header hoặc dòng bắt đầu dự án
        raw_blocks = re.split(
            r'(?:\n#{1,3}\s*|\n\*\*\s*|\n(?=[A-Z0-9][a-zA-Z0-9\s–—\-]{3,60}\s*(?:\||\n\s*(?:Technologies|Công nghệ|Tools|Hardware))))',
            "\n" + proj_text
        )

        all_links = list(set(URL_PATTERN.findall(proj_text)))
        all_metrics = list(set(IMPACT_PATTERN.findall(proj_text)))

        INVALID_TITLE_KEYWORDS = {
            "projects", "personal projects", "featured projects", "academic projects",
            "dự án", "dự án tiêu biểu", "extracurricular", "extracurricular activities",
            "hoạt động", "team size", "technologies", "công nghệ", "tools", "responsibilities"
        }

        # 2. Ghép tiêu đề đứng riêng lẻ với block nội dung mô tả ngay sau nó (xử lý lỗi ngắt đoạn OCR/Docling)
        merged_blocks: List[str] = []
        skip_next = False

        for i in range(len(raw_blocks)):
            if skip_next:
                skip_next = False
                continue

            current_b = raw_blocks[i].strip()
            if not current_b:
                continue

            # Nếu block hiện tại chỉ là 1 dòng tiêu đề ngắn (< 75 chars) và block sau chứa nội dung chi tiết
            if i + 1 < len(raw_blocks):
                next_b = raw_blocks[i + 1].strip()
                if len(current_b.splitlines()) == 1 and len(current_b) < 75 and not DATE_SPAN_REGEX.search(current_b):
                    merged_blocks.append(f"{current_b}\n{next_b}")
                    skip_next = True
                    continue

            merged_blocks.append(current_b)

        projects: List[ProjectItem] = []

        # 3. Phân tích chi tiết từng block dự án
        for b in merged_blocks:
            if len(b) < 35:
                continue

            lines = [l.strip() for l in b.splitlines() if l.strip()]
            if not lines:
                continue

            # Trích xuất và chuẩn hóa tên dự án
            raw_title = re.sub(r'^[#*_\-\s]+', '', lines[0]).strip('|-–— :*')
            title = re.split(r'[|–—]', raw_title)[0].strip()
            title = DATE_SPAN_REGEX.sub('', title).strip('()-–— :*')
            title_low = title.lower()

            # Bỏ qua nếu tiêu đề là mốc thời gian hoặc từ khóa mục chung
            if (
                len(title) < 3
                or title_low in INVALID_TITLE_KEYWORDS
                or (DATE_SPAN_REGEX.search(raw_title) and len(title.split()) <= 1)
            ):
                continue

            # Trích xuất Technologies của riêng block này
            tech_match = re.search(r'(?:Technologies|Công nghệ|Tools|Hardware|Stack)[:\s]+([^\n]+)', b, re.IGNORECASE)
            proj_techs: List[str] = []
            if tech_match:
                raw_tech = tech_match.group(1)
                proj_techs = [
                    t.strip().strip("•-* :.#▪\"'")
                    for t in re.split(r'[,|/]+', raw_tech)
                    if 1 < len(t.strip()) < 30 and not DATE_SPAN_REGEX.search(t)
                ]

            # Fallback scan kỹ năng nếu không có dòng Technologies riêng
            if not proj_techs:
                proj_techs = [
                    s for s in detected_skills 
                    if re.search(r'\b' + re.escape(s) + r'\b', b, re.IGNORECASE)
                ][:6]

            proj_metrics = [m for m in all_metrics if m in b]
            proj_links = [l for l in all_links if l in b]

            # Nhận diện vai trò
            role = "Developer"
            if re.search(r'\b(lead|leader|trưởng nhóm)\b', b, re.IGNORECASE):
                role = "Leader"
            elif re.search(r'\b(solo|tự thực hiện|cá nhân)\b', b, re.IGNORECASE):
                role = "Solo Developer"

            score = cls.calculate_heuristic_score(b, proj_techs, proj_metrics)

            projects.append(ProjectItem(
                name=title[:60],
                role=role,
                technologies=proj_techs,
                description=b[:900],
                impact_metrics=proj_metrics,
                links=proj_links,
                heuristic_score=score
            ))

        return projects[:6]

    @classmethod
    def extract_education(cls, edu_text: str, full_text: str) -> Tuple[Optional[DegreeType], Optional[str], List[EducationItem]]:
        """Trích xuất chi tiết trường học, ngành học, mốc năm và GPA."""
        degree: Optional[DegreeType] = None
        target_text = edu_text if edu_text.strip() else full_text
        text_lower = target_text.lower()

        if re.search(r'\b(tiến sĩ|ph\.d|doctorate)\b', text_lower):
            degree = "Ph.D"
        elif re.search(r'\b(thạc sĩ|master)\b', text_lower):
            degree = "Master"
        elif re.search(r'\b(kỹ sư|engineer)\b', text_lower):
            degree = "Engineer"
        elif re.search(r'\b(cử nhân|bachelor|b\.s|b\.a|bs|ba)\b', text_lower):
            degree = "Bachelor"
        elif re.search(r'\b(cao đẳng|associate)\b', text_lower):
            degree = "Associate"

        # Match chuyên ngành
        major = None
        major_match = re.search(r'(?:major|chuyên ngành|ngành|bachelor of|degree in)[:\s]+([a-zA-Zà-ỹÀ-Ỹ\s&]{3,40})', target_text, re.IGNORECASE)
        if major_match:
            raw_major = major_match.group(1).strip()
            if embedder is not None and len(raw_major) > 4:
                m_emb = embedder.encode(raw_major, convert_to_tensor=True)
                c_embs = embedder.encode(CANONICAL_FIELDS, convert_to_tensor=True)
                scores = util.cos_sim(m_emb, c_embs)[0]
                best_idx = int(scores.argmax())
                if scores[best_idx] > 0.45:
                    major = CANONICAL_FIELDS[best_idx]
            else:
                major = raw_major

        # Trích xuất trường & GPA
        edu_history: List[EducationItem] = []
        inst_match = re.search(r'(?:University|Trường Đại học|Trường Cao đẳng|Đại học|Cao đẳng|Institute|Academy)[^\n|,|;]+', target_text, re.IGNORECASE)
        inst_name = inst_match.group(0).strip(' -#*') if inst_match else None

        gpa_match = re.search(r'(?:GPA|Điểm|Grade)[:\s]*([0-9\.]+(?:\s*/\s*[0-9\.]+)?|\b[0-9\.]+\s*/\s*10\b)', target_text, re.IGNORECASE)
        gpa_str = gpa_match.group(0).strip() if gpa_match else None

        yr_match = re.search(r'\b(20\d{2})\b.*[-–—tođến\s].*\b(20\d{2})\b', target_text)
        s_yr = int(yr_match.group(1)) if yr_match else None
        e_yr = int(yr_match.group(2)) if yr_match else None

        if inst_name or major:
            edu_history.append(EducationItem(
                institution=inst_name or "University Institution",
                degree=degree,
                field_of_study=major,
                start_year=s_yr,
                end_year=e_yr,
                gpa_or_honors=gpa_str
            ))

        return degree, major, edu_history

    @classmethod
    def extract_skills(cls, skills_text: str, full_text: str) -> List[str]:
        """Lọc và chuẩn hóa token kỹ năng, loại bỏ hoàn toàn markdown header và date noise."""
        raw_source = skills_text if len(skills_text.strip()) > 15 else full_text
        match = re.search(r'(?:Skills|Kỹ năng|Technical Skills)[\s:]*\n(.*?)(?:\n[A-ZÀ-Ỹ\s]{4,}|\Z)', raw_source, re.IGNORECASE | re.DOTALL)
        content_to_split = match.group(1) if match else raw_source

        raw_tokens = re.split(r'[,•\n\t|/]+', content_to_split)
        clean_skills = []

        for t in raw_tokens:
            token = t.strip().strip("-* :.#•▪\"'")
            token_low = token.lower()
            # Loại bỏ headers, noise, ngày tháng, câu mô tả dài
            if len(token) < 2 or len(token) > 35:
                continue
            if token.startswith("##") or any(kw in token_low for kw in ["language", "working environment", "coach at", "intern at", "oct.", "now", "present", "2025", "2024", "gpa", "toeic", "ielts", "programming:"]):
                continue
            if not any(token_low == s.lower() for s in clean_skills):
                clean_skills.append(token)

        return clean_skills[:25]

    @classmethod
    def parse(cls, full_text: str) -> StructuredResume:
        sections = cls.isolate_sections(full_text)
        skills = cls.extract_skills(sections["skills"], full_text)
        exp_years, job_titles, work_exp = cls.extract_work_experience(sections["experience"])
        projects = cls.extract_projects(sections["projects"], skills)
        degree, field_of_study, edu_history = cls.extract_education(sections["education"], full_text)

        certifications = [
            c.strip() for c in re.split(r'[,•\n]+', sections["certifications"])
            if 3 < len(c.strip()) < 50 and not c.strip().startswith("##")
        ]

        return StructuredResume(
            summary=sections["summary"][:400].strip() if sections["summary"] else None,
            education_degree=degree,
            education_field=field_of_study,
            education_history=edu_history,
            experience_years=exp_years,
            job_titles=job_titles,
            work_experience=work_exp,
            projects=projects,
            skills=skills,
            certifications=certifications
        )


# ---------------------------------------------------------------------------
# 5. ONLINE LLM STRUCTURING ENGINE (GROQ)
# ---------------------------------------------------------------------------
def parse_resume_llm(text: str, client: Groq) -> StructuredResume:
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
    print(f"Input Cleaned   : {INPUT_CLEANED_TEXT}")
    print(f"Output Parsed   : {OUTPUT_PARSED_JSON}")
    print("=" * 80)

    if not INPUT_CLEANED_TEXT.exists():
        print(f"❌ Error: {INPUT_CLEANED_TEXT} not found. Please run main.py (Stage 1) first.")
        return

    with open(INPUT_CLEANED_TEXT, "r", encoding="utf-8") as f:
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