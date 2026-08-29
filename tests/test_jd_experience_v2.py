"""R1.2 tests: JD Experience Extractor V2 — section-aware extraction."""
from __future__ import annotations

import sys
from pathlib import Path

_data_loader_dir = str(Path(__file__).resolve().parent.parent / "src" / "Data_loader")
if _data_loader_dir not in sys.path:
    sys.path.insert(0, _data_loader_dir)

from offline.jd_offline_parser import (
    detect_sections,
    extract_min_experience_v2,
)


def _exp(text: str) -> float:
    """Helper: run detect_sections + extract_min_experience_v2 on raw text."""
    sections = detect_sections(text)
    return extract_min_experience_v2(sections=sections, full_text=text)


# -----------------------------------------------------------------------
# 1. Standard English
# -----------------------------------------------------------------------
def test_standard_english():
    text = """
Requirements
- At least 2 years of experience in machine learning.
"""
    assert _exp(text) == 2.0


# -----------------------------------------------------------------------
# 2. Plus format
# -----------------------------------------------------------------------
def test_plus_format():
    text = """
Requirements
- 3+ years of software development experience.
"""
    assert _exp(text) == 3.0


# -----------------------------------------------------------------------
# 3. Range format (hyphen and en-dash)
# -----------------------------------------------------------------------
def test_range_hyphen():
    text = """
Requirements
- 1-2 years of experience.
"""
    assert _exp(text) == 1.0


def test_range_en_dash():
    text = """
Requirements
- 1 – 2 years experience.
"""
    assert _exp(text) == 1.0


# -----------------------------------------------------------------------
# 4. Vietnamese
# -----------------------------------------------------------------------
def test_vietnamese():
    text = """
YÊU CẦU ỨNG VIÊN
- Tối thiểu 2 năm kinh nghiệm phát triển phần mềm.
"""
    assert _exp(text) == 2.0


# -----------------------------------------------------------------------
# 5. Vietnamese range
# -----------------------------------------------------------------------
def test_vietnamese_range():
    text = """
YÊU CẦU
- 1-3 năm kinh nghiệm.
"""
    assert _exp(text) == 1.0


# -----------------------------------------------------------------------
# 6. Months
# -----------------------------------------------------------------------
def test_months():
    text = """
Requirements
- At least 6 months of experience.
"""
    assert _exp(text) == 0.5


# -----------------------------------------------------------------------
# 7. Experience section bare value
# -----------------------------------------------------------------------
def test_experience_section_bare_value():
    text = """
Experience
- 2+ years
"""
    assert _exp(text) == 2.0


# -----------------------------------------------------------------------
# 8. Requirements short bare value
# -----------------------------------------------------------------------
def test_requirements_short_bare_value():
    text = """
Requirements
- Minimum 1 year
"""
    assert _exp(text) == 1.0


# -----------------------------------------------------------------------
# 9. No experience required
# -----------------------------------------------------------------------
def test_no_experience_required():
    text = """
Requirements
- No prior experience required.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# 10. Fresh graduate
# -----------------------------------------------------------------------
def test_fresh_graduate():
    text = """
Requirements
- Fresh graduates are welcome.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# 11. Intern title must NOT override numeric
# -----------------------------------------------------------------------
def test_intern_title_no_override():
    text = """
AI Engineer Intern

Requirements
- 1 year of experience with Python.
"""
    assert _exp(text) == 1.0


# -----------------------------------------------------------------------
# 12. Multiple mandatory requirements → max
# -----------------------------------------------------------------------
def test_multiple_mandatory_max():
    text = """
Requirements
- 2 years of software engineering experience.
- 1 year of machine learning experience.
"""
    assert _exp(text) == 2.0


# -----------------------------------------------------------------------
# 13. Preferred must NOT override required
# -----------------------------------------------------------------------
def test_preferred_no_override():
    text = """
Requirements
- 1 year of experience.

Preferred Qualifications
- 3 years of cloud experience.
"""
    assert _exp(text) == 1.0


# -----------------------------------------------------------------------
# 14. Preferred only → 0.0
# -----------------------------------------------------------------------
def test_preferred_only():
    text = """
Preferred Qualifications
- 3 years of AWS experience.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# 15. Company age false positive
# -----------------------------------------------------------------------
def test_company_age_false_positive():
    text = """
Overview
Our company has over 10 years of experience in the market.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# 16. Project duration false positive
# -----------------------------------------------------------------------
def test_project_duration_false_positive():
    text = """
Responsibilities
You will work on a 3-year project.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# 17. Education duration false positive
# -----------------------------------------------------------------------
def test_education_duration_false_positive():
    text = """
Requirements
Bachelor degree from a 4-year program.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# 18. Date false positive
# -----------------------------------------------------------------------
def test_date_false_positive():
    text = """
Overview
Founded in 2024.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# 19. Decimal years
# -----------------------------------------------------------------------
def test_decimal_years():
    text = """
Requirements
- 0.5 years of relevant experience.
"""
    assert _exp(text) == 0.5


# -----------------------------------------------------------------------
# 20. No evidence
# -----------------------------------------------------------------------
def test_no_evidence():
    text = """
Requirements
- Strong Python skills.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# Additional: Vietnamese zero experience
# -----------------------------------------------------------------------
def test_vietnamese_zero_experience():
    text = """
YÊU CẦU
- Không yêu cầu kinh nghiệm.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# Additional: "to" range
# -----------------------------------------------------------------------
def test_range_to():
    text = """
Requirements
- 2 to 4 years of experience.
"""
    assert _exp(text) == 2.0


# -----------------------------------------------------------------------
# Additional: Vietnamese "đến" range
# -----------------------------------------------------------------------
def test_vietnamese_range_den():
    text = """
YÊU CẦU
- 1 đến 2 năm kinh nghiệm.
"""
    assert _exp(text) == 1.0


# -----------------------------------------------------------------------
# Additional: 12 months → 1.0
# -----------------------------------------------------------------------
def test_twelve_months():
    text = """
Requirements
- Minimum 12 months of experience.
"""
    assert _exp(text) == 1.0


# -----------------------------------------------------------------------
# Additional: "over/more than" prefix
# -----------------------------------------------------------------------
def test_over_prefix():
    text = """
Requirements
- Over 3 years of relevant experience.
"""
    assert _exp(text) == 3.0


def test_more_than_prefix():
    text = """
Requirements
- More than 2 years of professional experience.
"""
    assert _exp(text) == 2.0


# -----------------------------------------------------------------------
# Additional: Organization false positive
# -----------------------------------------------------------------------
def test_organization_experience_false_positive():
    text = """
Overview
The organization has more than 20 years of experience.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# Less-than qualifier: literal <
# -----------------------------------------------------------------------
def test_less_than_literal():
    text = """
Requirements
- Recent graduate with <1 year of experience.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# Less-than qualifier: HTML-encoded &lt;
# -----------------------------------------------------------------------
def test_less_than_html_encoded():
    text = """
Requirements
- Recent graduate with &lt;1 year of experience.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# Less-than qualifier: Vietnamese "dưới"
# -----------------------------------------------------------------------
def test_less_than_vietnamese():
    text = """
YÊU CẦU
- Ứng viên có dưới 1 năm kinh nghiệm.
"""
    assert _exp(text) == 0.0


# -----------------------------------------------------------------------
# Less-than must NOT override strong mandatory numeric evidence
# -----------------------------------------------------------------------
def test_less_than_does_not_override_mandatory():
    text = """
Requirements
- At least 1 year of Python experience.
- Candidates with <2 years of industry experience may apply.
"""
    assert _exp(text) == 1.0
