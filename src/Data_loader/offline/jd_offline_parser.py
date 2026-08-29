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
# EXPERIENCE V2 — section-aware candidate experience extraction
# ---------------------------------------------------------------------------
# NOTE: The existing schema collapses "no reliable evidence" and "explicitly
# zero experience required" both into 0.0.  This is a known limitation of the
# current StructuredJobDescription.min_experience_years design.
# ---------------------------------------------------------------------------

# --- Compiled regexes (deterministic, no MiniLM) ---

# Unit detection: extracts lower-bound number + unit (years/months/năm/tháng)
_EXP_UNIT_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*\+?\s*'
    r'(?:[-–—]\s*|\s*(?:to|đến)\s+)?'
    r'(?:\d+(?:\.\d+)?\s*\+?\s*)?'
    r'(năm|years?|yrs?|months?|mos?|tháng)',
    re.IGNORECASE,
)

# Less-than experience qualifiers.
# Matches forms like:  <1 year,  &lt;1 year,  less than 1 year,
#   under 1 year,  dưới 1 năm,  ít hơn 1 năm,  chưa đến 1 năm
# When matched, the line expresses that candidates with LESS THAN the
# stated duration are accepted — i.e. the minimum required is effectively 0.
_LESS_THAN_EXP_RE = re.compile(
    r'(?:<|&lt;|&amp;lt;|less\s+than|under|dưới|ít\s+hơn|chưa\s+đến)'
    r'\s*'
    r'\d+(?:\.\d+)?\s*\+?\s*'
    r'(?:[-–—]\s*|\s*(?:to|đến)\s+)?'
    r'(?:\d+(?:\.\d+)?\s*\+?\s*)?'
    r'(?:năm|years?|yrs?|months?|mos?|tháng)',
    re.IGNORECASE,
)

# Candidate-experience cue words — the line must contain at least one of these
# for the numeric match to be considered genuine candidate experience.
# This guards against company-age, project-duration, and education-duration
# false positives.
_EXP_CUE_RE = re.compile(
    r'\b(?:experience|kinh\s+nghiệm|minimum|at\s+least|required|'
    r'tối\s+thiểu|ít\s+nhất|yêu\s+cầu|cần\s+có|'
    r'professional|hands-on|work\s+experience|prior\s+experience|'
    r'relevant|related)\b',
    re.IGNORECASE,
)

# Explicit zero-experience / fresher cues
_ZERO_EXP_RE = re.compile(
    r'\b(?:no\s+(?:prior\s+)?experience\s+(?:is\s+)?required|'
    r'experience\s+is\s+not\s+required|'
    r'fresh\s+graduates?\s+(?:are\s+)?welcome|'
    r'fresh\s+graduate|'
    r'fresher(?:s)?(?:\s+(?:accepted|welcome))?|'
    r'không\s+(?:yêu\s+cầu|cần)\s+kinh\s+nghiệm|'
    r'chấp\s+nhận\s+fresher|'
    r'sinh\s+viên\s+mới\s+tốt\s+nghiệp|'
    r'mới\s+tốt\s+nghiệp\s+được\s+chấp\s+nhận)\b',
    re.IGNORECASE,
)

# False-positive blockers: lines matching these patterns should NOT be
# treated as candidate experience even if they contain a numeric duration.
_EXP_FALSE_POSITIVE_RE = re.compile(
    r'\b(?:company\s+has|organization\s+has|founded|'
    r'project\s+duration|contract\s+(?:is\s+)?valid|'
    r'year\s+program|year\s+bachelor|year\s+degree|'
    r'year\s+project)\b',
    re.IGNORECASE,
)


# Strict regex for bare duration lines.
# A bare duration line is essentially *only* a duration expression,
# e.g., "2+ years", "Minimum 2 years", "6 months", "1-2 years".
# It should NOT match "3-year contract" or "4-year degree".
_BARE_DURATION_RE = re.compile(
    r'^(?:[•\-*+]\s+)?'
    r'(?:minimum\s+(?:of\s+)?|at\s+least\s+|over\s+|more\s+than\s+|from\s+|'
    r'tối\s+thiểu\s+|ít\s+nhất\s+|từ\s+|trên\s+|hơn\s+)?'
    r'\d+(?:\.\d+)?\s*\+?\s*'
    r'(?:[-–—]\s*|\s*(?:to|đến)\s+)?'
    r'(?:\d+(?:\.\d+)?\s*\+?\s*)?'
    r'(?:năm|years?|yrs?|months?|mos?|tháng)'
    r'(?:\s*\+)?'
    r'[.\s]*$',
    re.IGNORECASE,
)


def _is_less_than_experience(line: str, match_text: str) -> bool:
    """Check if the matched duration is prefixed by a less-than qualifier.

    Instead of searching the entire line, we check if the less-than
    qualifier specifically qualifies our matched unit text.
    """
    # Find all less-than expressions in the line
    for lt_match in _LESS_THAN_EXP_RE.finditer(line):
        if match_text in lt_match.group(0):
            return True
    return False


def _parse_experience_durations(line: str) -> List[float]:
    """Parse a single line for all numeric experience durations.

    Returns a list of candidate minimum years.

    - **Less-than qualifiers** (``<N``, ``&lt;N``, ``less than N``) are
      omitted because they express that candidates with less than
      the stated duration are accepted (i.e. not a mandatory minimum).
    - Ranges use the **lower** bound (e.g. ``1-2 years`` → ``1.0``).
    - Months are converted to years (e.g. ``6 months`` → ``0.5``).
    - Results are rounded to one decimal place.
    """
    durations = []
    for m in _EXP_UNIT_RE.finditer(line):
        match_text = m.group(0)
        
        # Less-than qualifier → omit this duration as it's not a minimum
        if _is_less_than_experience(line, match_text):
            continue

        number = float(m.group(1))
        unit = m.group(2).lower()

        is_months = unit in ("month", "months", "mo", "mos", "tháng")

        if is_months:
            durations.append(round(number / 12.0, 1))
        else:
            durations.append(round(number, 1))
            
    return durations


def _is_zero_experience_cue(line: str) -> bool:
    """Check if a line explicitly states no experience is required."""
    return bool(_ZERO_EXP_RE.search(line))


def _has_experience_cue(line: str) -> bool:
    """Check if a line contains candidate-experience context words."""
    return bool(_EXP_CUE_RE.search(line))


def _is_false_positive(line: str) -> bool:
    """Check if a line matches known false-positive patterns."""
    return bool(_EXP_FALSE_POSITIVE_RE.search(line))


def _is_bare_duration_line(line: str) -> bool:
    """Check if a line is a short, bare duration statement.

    Examples of bare duration lines::

        "2+ years"
        "Minimum 2 years"
        "6 months"
        "At least 1 year"
        "Tối thiểu 2 năm"

    These are acceptable inside experience/requirements sections even
    without a full ``experience`` cue word.
    """
    return bool(_BARE_DURATION_RE.match(line.strip()))


def extract_min_experience_v2(
    sections: Dict[str, List[str]],
    full_text: str,
) -> float:
    """Extract minimum required candidate experience using section context.

    Algorithm
    ---------
    1. Collect valid mandatory numeric evidence from **experience** and
       **requirements** sections (Tier 1 & 2).
    2. If numeric mandatory candidates exist → return ``max(candidates)``.
    3. Otherwise, check guarded numeric fallback in **overview** / **other**
       (Tier 3). Accept only if the line has an explicit experience cue.
    4. If no numeric evidence → evaluate explicit zero-experience / fresher
       cues across Tier 1–3.
    5. If still no evidence → return ``0.0`` (schema default; means
       "unknown / no reliable evidence").

    Parameters
    ----------
    sections : Dict[str, List[str]]
        Output from ``detect_sections()``.
    full_text : str
        The original cleaned JD text (used only if all sections are empty).
    """
    mandatory_candidates: List[float] = []

    # --- TIER 1: experience section ---
    for line in sections.get("experience", []):
        if _is_false_positive(line):
            continue
        durs = _parse_experience_durations(line)
        if durs:
            # Inside a dedicated experience section, accept even bare
            # duration lines without explicit cue words.
            mandatory_candidates.extend(durs)

    # --- TIER 2: requirements section ---
    for line in sections.get("requirements", []):
        if _is_false_positive(line):
            continue
        durs = _parse_experience_durations(line)
        if durs:
            if _has_experience_cue(line) or _is_bare_duration_line(line):
                mandatory_candidates.extend(durs)

    # Step 2: if we found mandatory numeric evidence, return max
    if mandatory_candidates:
        return max(mandatory_candidates)

    # --- TIER 3: guarded fallback in overview / other ---
    for section_key in ("overview", "other"):
        for line in sections.get(section_key, []):
            if _is_false_positive(line):
                continue
            durs = _parse_experience_durations(line)
            if durs and _has_experience_cue(line):
                mandatory_candidates.extend(durs)

    if mandatory_candidates:
        return max(mandatory_candidates)

    # Step 4: explicit zero-experience cues
    tier_keys = ("experience", "requirements", "overview", "other")
    for key in tier_keys:
        for line in sections.get(key, []):
            if _is_zero_experience_cue(line):
                return 0.0

    # Step 5: no reliable evidence → 0.0 (schema default)
    return 0.0


# ---------------------------------------------------------------------------
# OFFLINE HEURISTIC JD EXTRACTOR (REGEX + MiniLM)
# ---------------------------------------------------------------------------
class OfflineJDExtractor:
    @staticmethod
    def extract_min_experience(text: str) -> float:
        """Legacy V1 experience extraction (kept for backward compatibility)."""
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

        # R1.1 section map — computed once, reused for extraction
        sections = detect_sections(text)

        req_skills, responsibilities, certs = cls.extract_skills_and_responsibilities(text)
        exp_years = extract_min_experience_v2(sections=sections, full_text=text)
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

