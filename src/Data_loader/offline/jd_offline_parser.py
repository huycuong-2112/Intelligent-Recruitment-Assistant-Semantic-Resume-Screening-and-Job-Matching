from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

try:
    from ..jd_schema import DegreeType, StructuredJobDescription
except ImportError:
    from jd_schema import DegreeType, StructuredJobDescription


# ---------------------------------------------------------------------------
# CANONICAL INTERNAL SECTION CATEGORIES
# ---------------------------------------------------------------------------
CANONICAL_SECTIONS = (
    "overview",
    "responsibilities",
    "requirements",
    "preferred",
    "education",
    "experience",
    "skills",
    "certifications",
    "benefits",
    "other",
)

# ---------------------------------------------------------------------------
# SECTION ALIASES — bilingual heading → canonical category mapping
# ---------------------------------------------------------------------------
# Each alias is stored in lowercase. Lookup is done against a normalized
# heading string (see ``normalize_heading``).
#
# Combined-heading policy (documented):
#   "requirements / qualifications" → requirements
#   "skills & qualifications"       → requirements
#   "education & experience"        → requirements
# Rationale: combined sections usually describe candidate eligibility
# requirements, not a single isolated field.
# ---------------------------------------------------------------------------
JD_SECTION_ALIASES: Dict[str, str] = {}

_ALIAS_GROUPS: Dict[str, List[str]] = {
    "overview": [
        "overview",
        "job overview",
        "about the role",
        "about this role",
        "role overview",
        "position overview",
        "job description",
        # Vietnamese
        "mô tả chung",
        "tổng quan",
        "giới thiệu công việc",
        "giới thiệu vị trí",
    ],
    "responsibilities": [
        "responsibilities",
        "job responsibilities",
        "key responsibilities",
        "duties",
        "job duties",
        "what you will do",
        "what you'll do",
        "what you\u2019ll do",
        "your responsibilities",
        # Vietnamese
        "nhiệm vụ",
        "trách nhiệm",
        "trách nhiệm công việc",
        "công việc chính",
        "mô tả công việc",
    ],
    "requirements": [
        "requirements",
        "job requirements",
        "qualifications",
        "minimum qualifications",
        "required qualifications",
        "candidate requirements",
        "what we are looking for",
        "what we're looking for",
        "what we\u2019re looking for",
        # Vietnamese
        "yêu cầu",
        "yêu cầu công việc",
        "yêu cầu ứng viên",
        "tiêu chí",
        "điều kiện",
        # Combined headings → requirements
        "requirements / qualifications",
        "requirements/qualifications",
        "skills & qualifications",
        "skills and qualifications",
        "education & experience",
        "education and experience",
    ],
    "preferred": [
        "preferred",
        "preferred qualifications",
        "preferred skills",
        "nice to have",
        "nice-to-have",
        "good to have",
        "bonus",
        "plus",
        "advantage",
        "preferred requirements",
        # Vietnamese
        "ưu tiên",
        "điểm cộng",
        "lợi thế",
        "là một lợi thế",
    ],
    "education": [
        "education",
        "educational requirements",
        "academic requirements",
        "degree",
        "degree requirements",
        # Vietnamese
        "học vấn",
        "trình độ học vấn",
        "bằng cấp",
        "yêu cầu bằng cấp",
    ],
    "experience": [
        "experience",
        "work experience",
        "experience requirements",
        "professional experience",
        # Vietnamese
        "kinh nghiệm",
        "kinh nghiệm làm việc",
        "yêu cầu kinh nghiệm",
    ],
    "skills": [
        "skills",
        "technical skills",
        "required skills",
        "core skills",
        "competencies",
        "technical requirements",
        # Vietnamese
        "kỹ năng",
        "kỹ năng chuyên môn",
        "kỹ năng kỹ thuật",
        "năng lực",
    ],
    "certifications": [
        "certifications",
        "certification",
        "certificates",
        "licenses",
        "professional certifications",
        # Vietnamese
        "chứng chỉ",
        "chứng nhận",
        "giấy phép",
    ],
    "benefits": [
        "benefits",
        "compensation",
        "perks",
        "what we offer",
        "employee benefits",
        # Vietnamese
        "quyền lợi",
        "phúc lợi",
        "chế độ",
        "đãi ngộ",
    ],
}

# Build the flat lookup dict
for _category, _aliases in _ALIAS_GROUPS.items():
    for _alias in _aliases:
        JD_SECTION_ALIASES[_alias] = _category


# ---------------------------------------------------------------------------
# HEADING NORMALIZATION
# ---------------------------------------------------------------------------
# Regex to strip markdown heading markers, emphasis, bullets, and numbering prefixes
_HEADING_PREFIX_RE = re.compile(
    r'^(?:'
    r'#{1,6}\s*'           # Markdown headings: # ## ### etc.
    r'|\*{1,2}\s*'         # Bold/emphasis markers: * **
    r'|_{1,2}\s*'          # Underscores: _ __
    r'|[-–—=]{2,}\s*'      # Separator lines: -- === ---
    r'|[•\-*+]\s+'         # Bullets: • - * +
    r'|\d{1,2}[.)]\s*'     # Numbering: 1. 2) 3.
    r'|[IVXLC]{1,4}[.)]\s*'  # Roman numerals: I. II) III.
    r')*',
    re.IGNORECASE,
)


def normalize_heading(raw_line: str) -> str:
    """Normalize a candidate heading line to a comparable lowercase string.

    Strips markdown markers, emphasis, bullets, numbering, trailing colons,
    surrounding separators, and normalizes whitespace/case.

    Examples::

        "## REQUIREMENTS:"          → "requirements"
        "**Requirements**"          → "requirements"
        "### Responsibilities"      → "responsibilities"
        "1. Requirements"           → "requirements"
        "2) Responsibilities"       → "responsibilities"
        "III. Benefits"             → "benefits"
        "• REQUIREMENTS"            → "requirements"
        "YÊU CẦU CÔNG VIỆC:"       → "yêu cầu công việc"
        "--- Preferred Qualifications ---" → "preferred qualifications"
    """
    text = _HEADING_PREFIX_RE.sub('', raw_line.strip())
    # Strip trailing colons, markdown, spaces, etc.
    text = text.strip(" *_-=:#•.,\t")
    # Normalize Unicode apostrophes → ASCII for alias matching
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


# ---------------------------------------------------------------------------
# HEADING CLASSIFICATION
# ---------------------------------------------------------------------------
# Max length for a line to be considered a candidate heading.
# Lines longer than this are almost certainly content sentences, not headings.
_MAX_HEADING_LEN = 60


def classify_heading(line: str) -> Optional[str]:
    """Classify a single line as a canonical section heading, or ``None``.

    Returns the canonical section name (e.g. ``"requirements"``) if the line
    is determined to be a section heading, or ``None`` if it is ordinary
    content.

    **False-positive protection:**
    - Lines longer than ``_MAX_HEADING_LEN`` chars (after stripping) are
      rejected outright — they are content sentences, not headings.
    - Only the *normalized full line* is matched against known aliases.
      Substring matching is intentionally avoided.
    - A line must look structurally heading-like (short, or has markdown
      markers / numbering / trailing colon) to be accepted.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Hard reject: long lines are content, not headings
    if len(stripped) > _MAX_HEADING_LEN:
        return None

    normalized = normalize_heading(stripped)
    if not normalized:
        return None

    # Exact alias match — the primary classification path
    if normalized in JD_SECTION_ALIASES:
        return JD_SECTION_ALIASES[normalized]

    # No match
    return None


# ---------------------------------------------------------------------------
# SECTION DETECTOR  —  main entry point
# ---------------------------------------------------------------------------
def detect_sections(cleaned_text: str) -> Dict[str, List[str]]:
    """Detect and split a cleaned JD text into canonical sections.

    Parameters
    ----------
    cleaned_text : str
        The ``content`` field from a record in ``cleaned_jds.json``.

    Returns
    -------
    Dict[str, List[str]]
        Keys are the canonical section names (see ``CANONICAL_SECTIONS``).
        Values are lists of content lines belonging to that section.
        All canonical keys are always present (possibly with empty lists).

    Behavior
    --------
    * **Preamble**: Text before the first detected heading goes to
      ``"overview"``.
    * **Repeated sections**: If the same heading appears multiple times,
      content is **appended** (not overwritten).
    * **No-heading fallback**: If no headings are found, the entire text
      goes to ``"other"``.
    * **Headings themselves** are excluded from body content.
    """
    sections: Dict[str, List[str]] = {key: [] for key in CANONICAL_SECTIONS}

    lines = cleaned_text.splitlines()
    current_section: Optional[str] = None  # None = preamble (before first heading)
    found_any_heading = False

    for line in lines:
        heading = classify_heading(line)

        if heading is not None:
            current_section = heading
            found_any_heading = True
            continue  # skip the heading line itself

        # Content line
        content = line.strip()
        if not content:
            continue

        if current_section is not None:
            sections[current_section].append(content)
        else:
            # Preamble: text before first heading → overview
            sections["overview"].append(content)

    # No-heading fallback: put everything in "other"
    if not found_any_heading:
        sections["overview"] = []  # undo any preamble placed there
        for line in lines:
            stripped = line.strip()
            if stripped:
                sections["other"].append(stripped)

    return sections


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
