"""Tests for the JD Section Detector introduced in R1.1."""
from __future__ import annotations

import sys
from pathlib import Path

# Support running from project root
_data_loader_dir = str(Path(__file__).resolve().parent.parent / "src" / "Data_loader")
if _data_loader_dir not in sys.path:
    sys.path.insert(0, _data_loader_dir)

from offline.jd_offline_parser import (
    classify_heading,
    detect_sections,
    normalize_heading,
)


def test_normalize_heading():
    assert normalize_heading("## REQUIREMENTS:") == "requirements"
    assert normalize_heading("**Requirements**") == "requirements"
    assert normalize_heading("### Responsibilities") == "responsibilities"
    assert normalize_heading("1. Requirements") == "requirements"
    assert normalize_heading("2) Responsibilities") == "responsibilities"
    assert normalize_heading("III. Benefits") == "benefits"
    assert normalize_heading("• REQUIREMENTS") == "requirements"
    assert normalize_heading("YÊU CẦU CÔNG VIỆC:") == "yêu cầu công việc"
    assert normalize_heading("--- Preferred Qualifications ---") == "preferred qualifications"


def test_classify_heading():
    # True positives
    assert classify_heading("Requirements") == "requirements"
    assert classify_heading("## Responsibilities") == "responsibilities"
    assert classify_heading("YÊU CẦU ỨNG VIÊN") == "requirements"

    # False positives (should return None)
    assert classify_heading("Candidates must satisfy the requirements of the project.") is None
    assert classify_heading("AWS certification is preferred.") is None
    assert classify_heading("You will be responsible for implementing APIs.") is None
    
    # Very long lines should be rejected
    long_line = "We are looking for someone with skills " + "and " * 20 + "more skills"
    assert classify_heading(long_line) is None


def test_standard_english_jd():
    text = """
About the Role
This is an awesome job.
Responsibilities
- Do this
- Do that
Requirements
- Python
- AWS
Nice to Have
- Docker
Benefits
- Health insurance
"""
    sections = detect_sections(text)
    assert sections["overview"] == ["This is an awesome job."]
    assert sections["responsibilities"] == ["- Do this", "- Do that"]
    assert sections["requirements"] == ["- Python", "- AWS"]
    assert sections["preferred"] == ["- Docker"]
    assert sections["benefits"] == ["- Health insurance"]
    # Empty ones
    assert sections["education"] == []
    assert sections["experience"] == []
    assert sections["skills"] == []


def test_vietnamese_jd():
    text = """
TỔNG QUAN
Làm việc tuyệt vời.
MÔ TẢ CÔNG VIỆC
- Phát triển hệ thống AI.
- Xây dựng pipeline dữ liệu.
YÊU CẦU ỨNG VIÊN
- Giỏi toán
KINH NGHIỆM
- 2 năm
TRÌNH ĐỘ HỌC VẤN
- Đại học
QUYỀN LỢI
- Bảo hiểm
"""
    sections = detect_sections(text)
    assert sections["overview"] == ["Làm việc tuyệt vời."]
    assert sections["responsibilities"] == ["- Phát triển hệ thống AI.", "- Xây dựng pipeline dữ liệu."]
    assert sections["requirements"] == ["- Giỏi toán"]
    assert sections["experience"] == ["- 2 năm"]
    assert sections["education"] == ["- Đại học"]
    assert sections["benefits"] == ["- Bảo hiểm"]


def test_markdown_and_numbered_headings():
    text = """
## Responsibilities
Code well.
**Requirements:**
Be smart.
### Preferred Qualifications
Know Docker.
1. Responsibilities
Wait no we already had that.
2. Requirements
Also repeat.
3. Benefits
Money.
"""
    sections = detect_sections(text)
    # Repeated sections append
    assert sections["responsibilities"] == ["Code well.", "Wait no we already had that."]
    assert sections["requirements"] == ["Be smart.", "Also repeat."]
    assert sections["preferred"] == ["Know Docker."]
    assert sections["benefits"] == ["Money."]


def test_repeated_sections():
    text = """
Requirements
Python

Requirements
Git
"""
    sections = detect_sections(text)
    assert sections["requirements"] == ["Python", "Git"]


def test_preamble_preservation():
    text = """
AI Engineer Intern
We are looking for an intern to join the AI team.
You will work on NLP systems.

Responsibilities
- Develop models.
"""
    sections = detect_sections(text)
    assert sections["overview"] == [
        "AI Engineer Intern",
        "We are looking for an intern to join the AI team.",
        "You will work on NLP systems."
    ]
    assert sections["responsibilities"] == ["- Develop models."]


def test_no_heading_fallback():
    text = """
We are hiring a Python developer with 2 years of experience...
Candidate should know FastAPI and PostgreSQL...
"""
    sections = detect_sections(text)
    assert sections["overview"] == []
    assert sections["other"] == [
        "We are hiring a Python developer with 2 years of experience...",
        "Candidate should know FastAPI and PostgreSQL..."
    ]


def test_requirements_vs_preferred():
    text = """
Requirements
Python

Nice to Have
Docker
"""
    sections = detect_sections(text)
    assert sections["requirements"] == ["Python"]
    assert sections["preferred"] == ["Docker"]


def test_combined_heading():
    text = """
Skills & Qualifications
- Python
"""
    sections = detect_sections(text)
    assert sections["requirements"] == ["- Python"]

