from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

try:
    from ..resume_schema import (
        DegreeType,
        EducationItem,
        ExperienceItem,
        ProjectItem,
        StructuredResume,
    )
except ImportError:
    from resume_schema import (
        DegreeType,
        EducationItem,
        ExperienceItem,
        ProjectItem,
        StructuredResume,
    )

# ---------------------------------------------------------------------------
# OPTIONAL DEPENDENCY: MiniLM for education-field semantic mapping
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util
    # Mô hình MiniLM siêu nhẹ (~80MB), tối ưu cho CPU
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    embedder = None

# ---------------------------------------------------------------------------
# CONSTANTS USED BY OFFLINE EXTRACTION
# ---------------------------------------------------------------------------
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
# ENHANCED OFFLINE RESUME PARSER (REGEX + MiniML HEURISTICS)
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
