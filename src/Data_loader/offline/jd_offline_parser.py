from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    from ..jd_schema import DegreeType, StructuredJobDescription
except ImportError:
    from jd_schema import DegreeType, StructuredJobDescription


# ---------------------------------------------------------------------------
# OFFLINE HEURISTIC JD EXTRACTOR (REGEX + MiniLM)
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
