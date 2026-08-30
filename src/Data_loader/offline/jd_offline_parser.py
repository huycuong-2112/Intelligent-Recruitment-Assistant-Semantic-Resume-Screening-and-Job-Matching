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
    r'\b(?:experience|kinh\s+nghiệm|work\s+experience|'
    r'professional\s+experience|prior\s+experience|'
    r'hands-on\s+experience|kinh\s+nghiệm\s+làm\s+việc|'
    r'kinh\s+nghiệm\s+thực\s+tế)\b',
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
# EDUCATION & FIELDS V2
# ---------------------------------------------------------------------------
_DEGREE_RE = re.compile(
    r'\b(?:'
    r'ph\.?d|doctorate|tiến\s+sĩ|'  # Ph.D
    r'master(?:\'s)?|thạc\s+sĩ|graduate\s+degree|'  # Master
    r'engineer(?:ing)?\s+degree|degree\s+of\s+engineer|bằng\s+kỹ\s+sư|tốt\s+nghiệp\s+kỹ\s+sư|'  # Engineer
    r'bachelor(?:\'s)?|đại\s+học|cử\s+nhân|university\s+degree|college\s+degree|undergraduate\s+degree|'  # Bachelor
    r'associate|cao\s+đẳng|'  # Associate
    r'high\s+school|thpt|trung\s+học\s+phổ\s+thông'  # High School
    r')\b',
    re.IGNORECASE
)

_DEGREE_MAP = {
    'ph.d': 'Ph.D', 'phd': 'Ph.D', 'doctorate': 'Ph.D', 'tiến sĩ': 'Ph.D',
    'master': 'Master', "master's": 'Master', 'thạc sĩ': 'Master', 'graduate degree': 'Master',
    'engineer degree': 'Engineer', 'engineering degree': 'Engineer', 'degree of engineer': 'Engineer', 'bằng kỹ sư': 'Engineer', 'tốt nghiệp kỹ sư': 'Engineer',
    'bachelor': 'Bachelor', "bachelor's": 'Bachelor', 'đại học': 'Bachelor', 'cử nhân': 'Bachelor', 'university degree': 'Bachelor', 'college degree': 'Bachelor', 'undergraduate degree': 'Bachelor',
    'associate': 'Associate', 'cao đẳng': 'Associate',
    'high school': 'High School', 'thpt': 'High School', 'trung học phổ thông': 'High School'
}

_DEGREE_HIERARCHY = {
    'High School': 1,
    'Associate': 2,
    'Bachelor': 3,
    'Engineer': 3,
    'Master': 4,
    'Ph.D': 5,
    'Any': 0,
    'Other': 0
}

_FIELD_ALIASES = {
    'khoa học máy tính': 'Computer Science',
    'công nghệ thông tin': 'Information Technology',
    'kỹ thuật phần mềm': 'Software Engineering',
    'computer engineering': 'Computer Engineering',
    'trí tuệ nhân tạo': 'Artificial Intelligence',
    'khoa học dữ liệu': 'Data Science',
    'information systems': 'Information Systems',
    'cybersecurity': 'Cybersecurity',
    'an toàn thông tin': 'Information Security',
    'electrical engineering': 'Electrical Engineering',
    'electronics engineering': 'Electronics Engineering',
    'telecommunications': 'Telecommunications',
    'toán': 'Mathematics',
    'thống kê': 'Statistics',
    'cs': 'Computer Science',
    'ai': 'Artificial Intelligence',
    'ml': 'Machine Learning'
}

_PREFERRED_CUE_RE = re.compile(
    r'\b(?:preferred|desirable|nice\s*to\s*have|good\s*to\s*have|advantage|plus|bonus|ưu\s+tiên|lợi\s+thế|điểm\s+cộng)\b',
    re.IGNORECASE
)

_GENERIC_FIELD_RE = re.compile(
    r'\b(?:related\s+(?:field|discipline|major|fields)|relevant\s+(?:field|discipline|fields)|technical\s+(?:field|discipline)|similar\s+(?:field|discipline)|equivalent\s+(?:field|discipline))\b',
    re.IGNORECASE
)

_FIELD_PREFIX_RE = re.compile(
    r'\b(?:degree\s+in|bachelor\'s\s+in|master\'s\s+in|phd\s+in|major\s+in|background\s+in|academic\s+background\s+in|field\s+of\s+study|specialization\s+in|chuyên\s+ngành|ngành|tốt\s+nghiệp\s+ngành|được\s+đào\s+tạo\s+ngành)\s+(.+)',
    re.IGNORECASE
)

def _normalize_degree(match_str: str) -> DegreeType:
    s = match_str.lower().strip()
    for k, v in _DEGREE_MAP.items():
        if k in s:
            return v  # type: ignore
    return "Any"  # type: ignore

def _extract_degree_candidates(line: str, is_education_section: bool) -> List[Tuple[DegreeType, bool]]:
    candidates = []
    is_preferred = bool(_PREFERRED_CUE_RE.search(line))
    for m in _DEGREE_RE.finditer(line):
        deg = _normalize_degree(m.group(0))
        candidates.append((deg, not is_preferred))
    return candidates

def _resolve_required_degree(candidates: List[Tuple[DegreeType, bool]]) -> DegreeType:
    mandatory_candidates = [deg for deg, is_mandatory in candidates if is_mandatory]
    if not mandatory_candidates:
        return "Any"  # type: ignore
    min_deg = mandatory_candidates[0]
    min_rank = _DEGREE_HIERARCHY.get(min_deg, 0)
    for deg in mandatory_candidates[1:]:
        rank = _DEGREE_HIERARCHY.get(deg, 0)
        if rank < min_rank:
            min_deg = deg
            min_rank = rank
    return min_deg

def _extract_field_candidates(line: str) -> List[str]:
    fields = []
    m = _FIELD_PREFIX_RE.search(line)
    if m:
        text = m.group(1)
        tokens = re.split(r'[,/|]|(?:\bor\b)|(?:\band\b)|(?:\bhoặc\b)|(?:\bhay\b)', text)
        for t in tokens:
            t = t.strip().strip('.')
            if not t:
                continue
            if _GENERIC_FIELD_RE.search(t):
                continue
            lower_t = t.lower()
            if lower_t in _FIELD_ALIASES:
                fields.append(_FIELD_ALIASES[lower_t])
            elif 2 < len(t) < 40:
                fields.append(t)
    return fields

def extract_education_v2(sections: Dict[str, List[str]], full_text: str) -> Tuple[DegreeType, List[str]]:
    degree_candidates: List[Tuple[DegreeType, bool]] = []
    preferred_fields: List[str] = []

    # Tier 1 & 2
    for section_key in ("education", "requirements"):
        for line in sections.get(section_key, []):
            degree_candidates.extend(_extract_degree_candidates(line, is_education_section=(section_key == "education")))
            preferred_fields.extend(_extract_field_candidates(line))

    # Tier 3 fallback
    if not any(is_mandatory for _, is_mandatory in degree_candidates):
        for section_key in ("overview", "other"):
            for line in sections.get(section_key, []):
                if re.search(r'\b(?:education|degree|bằng|tốt nghiệp|chuyên ngành)\b', line, re.IGNORECASE):
                    degree_candidates.extend(_extract_degree_candidates(line, is_education_section=False))
                    preferred_fields.extend(_extract_field_candidates(line))

    # Preferred section
    for line in sections.get("preferred", []):
        for deg, _ in _extract_degree_candidates(line, is_education_section=False):
            degree_candidates.append((deg, False))
        preferred_fields.extend(_extract_field_candidates(line))

    dedup_fields = []
    for f in preferred_fields:
        if f not in dedup_fields:
            dedup_fields.append(f)

    return _resolve_required_degree(degree_candidates), dedup_fields


# ---------------------------------------------------------------------------
# SKILLS V2
# ---------------------------------------------------------------------------
_IT_SKILLS = [
    'Python', 'Java', 'C', 'C++', 'C#', 'JavaScript', 'TypeScript', 'SQL', 'Git', 'Linux', 'Docker',
    'Kubernetes', 'AWS', 'Azure', 'GCP', 'TensorFlow', 'PyTorch', 'scikit-learn', 'Pandas', 'NumPy',
    'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'REST API', 'FastAPI', 'Flask',
    'Django', 'Spark', 'Hadoop', 'Kafka', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'React', 'Node.js'
]

def extract_skills_v2(sections: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    req_skills = []
    pref_skills = []
    
    def extract_from_line(line: str) -> List[str]:
        found = []
        for s in _IT_SKILLS:
            pattern = r'\b' + re.escape(s)
            if s[-1].isalnum():
                pattern += r'\b'
            if re.search(pattern, line, re.IGNORECASE):
                found.append(s)
        return found

    for line in sections.get("skills", []) + sections.get("requirements", []):
        skills = extract_from_line(line)
        if _PREFERRED_CUE_RE.search(line):
            pref_skills.extend(skills)
        else:
            req_skills.extend(skills)

    for line in sections.get("preferred", []):
        pref_skills.extend(extract_from_line(line))

    def dedup(lst: List[str]) -> List[str]:
        res = []
        for x in lst:
            if x not in res:
                res.append(x)
        return res[:25]

    return dedup(req_skills), dedup(pref_skills)


# ---------------------------------------------------------------------------
# RESPONSIBILITIES & DELIVERABLES V2
# ---------------------------------------------------------------------------
_RESP_VERB_RE = re.compile(
    r'\b(?:develop|build|design|implement|maintain|manage|deploy|analyze|research|create|support|collaborate|'
    r'phát\s+triển|xây\s+dựng|thiết\s+kế|triển\s+khai|quản\s+lý|nghiên\s+cứu|phân\s+tích|vận\s+hành|hỗ\s+trợ|phối\s+hợp)\b',
    re.IGNORECASE
)

_DELIVERABLE_CUE_RE = re.compile(
    r'\b(?:models?|systems?|services?|apis?|pipelines?|platforms?|applications?|dashboards?|solutions?|products?|features?|modules?|reports?|'
    r'hệ\s+thống|mô\s+hình|ứng\s+dụng|nền\s+tảng|giải\s+pháp|sản\s+phẩm|tính\s+năng)\b',
    re.IGNORECASE
)

def extract_responsibilities_and_deliverables_v2(sections: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    responsibilities = []
    source_lines = sections.get("responsibilities", [])
    
    if source_lines:
        for line in source_lines:
            line_clean = line.strip().strip("•-* ")
            if len(line_clean) > 10 and not line_clean.startswith("#"):
                if line_clean not in responsibilities:
                    responsibilities.append(line_clean)
    else:
        for k in ("overview", "other"):
            for line in sections.get(k, []):
                line_clean = line.strip().strip("•-* ")
                if len(line_clean) > 10 and not line_clean.startswith("#") and _RESP_VERB_RE.search(line_clean):
                    if line_clean not in responsibilities:
                        responsibilities.append(line_clean)
                        
    responsibilities = responsibilities[:10]
    
    deliverables = []
    for r in responsibilities:
        if _DELIVERABLE_CUE_RE.search(r):
            deliverables.append(r)
    deliverables = deliverables[:5]
    
    return responsibilities, deliverables


# ---------------------------------------------------------------------------
# CERTIFICATIONS V2
# ---------------------------------------------------------------------------
_CERT_EXACT_RE = re.compile(r'\b(?:TOEIC|IELTS|PMP|CFA|JLPT|HSK)(?:\s+[\w\d.]+)?\b', re.IGNORECASE)
_CERT_CLOUD_RE = re.compile(r'\b(?:AWS|Azure|GCP|Google\s+Cloud)\b.*?\b(?:certified|certification|certificate|chứng\s+chỉ|chứng\s+nhận)\b', re.IGNORECASE)

def extract_certifications_v2(sections: Dict[str, List[str]]) -> List[str]:
    certs = []
    for k in ("certifications", "requirements", "preferred"):
        for line in sections.get(k, []):
            line_clean = line.strip().strip("•-* ")
            if _CERT_EXACT_RE.search(line_clean) or _CERT_CLOUD_RE.search(line_clean):
                if line_clean not in certs:
                    certs.append(line_clean)
    return certs[:5]


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

    @classmethod
    def parse(cls, text: str, fallback_title: str) -> StructuredJobDescription:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        
        # 1. Title Extraction
        title = fallback_title
        for l in lines[:10]:
            # Do not treat bullet points as the job title
            if l.lstrip().startswith(("-", "•", "*")):
                continue
                
            clean_l = l.strip("•-*_# ")
            if not clean_l:
                continue
                
            if len(clean_l) < 60 and not clean_l.isupper():
                category = classify_heading(normalize_heading(clean_l))
                if category is not None and category != "other":
                    continue
                title = clean_l
                break


        # 2. Company Name
        company_name = None
        for l in lines[:20]:
            m = re.search(r'^(?:Company(?: Name)?|Công ty|Doanh nghiệp):\s*(.+)$', l, re.IGNORECASE)
            if m:
                company_name = m.group(1).strip()
                break

        # 3. Detect Sections
        sections = detect_sections(text)

        # 4. Overview
        overview_lines = sections.get("overview", [])
        if overview_lines:
            overview = " ".join([l.strip().strip("•-* ") for l in overview_lines if len(l) > 10])
        else:
            overview = " ".join([l.strip().strip("•-* ") for l in lines[:15] if len(l) > 20 and not l.startswith("#")])
        overview = overview[:350].strip()

        # 5. Extract Details using V2 Extractor Logic
        exp_years = extract_min_experience_v2(sections=sections, full_text=text)
        req_degree, pref_fields = extract_education_v2(sections=sections, full_text=text)
        req_skills, pref_skills = extract_skills_v2(sections=sections)
        responsibilities, deliverables = extract_responsibilities_and_deliverables_v2(sections=sections)
        certs = extract_certifications_v2(sections=sections)

        if not title:
            title = fallback_title

        return StructuredJobDescription(
            job_title=title,
            company_name=company_name,
            job_overview=overview,
            min_experience_years=exp_years,
            required_degree=req_degree,
            preferred_fields=pref_fields,
            required_skills=req_skills,
            preferred_skills=pref_skills,
            responsibilities=responsibilities,
            key_deliverables=deliverables,
            required_certifications=certs
        )

