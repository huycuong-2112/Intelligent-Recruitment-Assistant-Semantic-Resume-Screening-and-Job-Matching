import pytest
from src.Data_loader.offline.jd_offline_parser import OfflineJDExtractor
from src.Data_loader.jd_schema import StructuredJobDescription

def _parse(text: str, fallback_title: str = "Test Title") -> StructuredJobDescription:
    return OfflineJDExtractor.parse(text, fallback_title)

# ---------------------------------------------------------------------------
# EDUCATION & ACADEMIC FIELDS
# ---------------------------------------------------------------------------
def test_education_ai_engineer_intern_not_engineer_degree():
    text = """
    AI Engineer Intern
    Requirements
    - Strong Python skills.
    """
    jd = _parse(text)
    assert jd.required_degree == "Any"

def test_education_bachelor_in_computer_science():
    text = """
    Requirements
    - Bachelor's degree in Computer Science.
    """
    jd = _parse(text)
    assert jd.required_degree == "Bachelor"
    assert "Computer Science" in jd.preferred_fields

def test_education_master_preferred_only():
    text = """
    Preferred Qualifications
    - Master's degree preferred.
    """
    jd = _parse(text)
    assert jd.required_degree == "Any"

def test_education_bachelor_required_master_preferred():
    text = """
    Requirements
    - Bachelor's degree required.
    - Master's degree preferred.
    """
    jd = _parse(text)
    assert jd.required_degree == "Bachelor"

def test_education_vietnamese_dai_hoc_cntt():
    text = """
    Yêu cầu ứng viên
    - Tốt nghiệp đại học chuyên ngành Công nghệ thông tin.
    """
    jd = _parse(text)
    assert jd.required_degree == "Bachelor"
    assert "Information Technology" in jd.preferred_fields

def test_education_explicit_engineer_degree():
    text = """
    Education
    - Engineer degree in Computer Engineering.
    """
    jd = _parse(text)
    assert jd.required_degree == "Engineer"
    assert "Computer Engineering" in jd.preferred_fields

def test_education_unknown_explicit_field_preserved():
    text = """
    Requirements
    - Bachelor's degree in Computational Linguistics.
    """
    jd = _parse(text)
    assert jd.required_degree == "Bachelor"
    assert "Computational Linguistics" in jd.preferred_fields

def test_education_related_field_removed():
    text = """
    Requirements
    - Degree in Computer Science or related field.
    """
    jd = _parse(text)
    assert "Computer Science" in jd.preferred_fields
    assert "related field" not in jd.preferred_fields

# ---------------------------------------------------------------------------
# SKILLS
# ---------------------------------------------------------------------------
def test_skills_required_python_git():
    text = """
    Requirements
    - Python and Git required.
    """
    jd = _parse(text)
    assert "Python" in jd.required_skills
    assert "Git" in jd.required_skills

def test_skills_preferred_docker():
    text = """
    Preferred
    - Experience with Docker is a plus.
    """
    jd = _parse(text)
    assert "Docker" in jd.preferred_skills
    assert "Docker" not in jd.required_skills

def test_skills_aws_experience():
    text = """
    Requirements
    - AWS experience required.
    """
    jd = _parse(text)
    assert "AWS" in jd.required_skills
    assert "AWS" not in jd.required_certifications

def test_skills_random_short_sentence_not_skill():
    text = """
    Requirements
    - Be a good team player.
    """
    jd = _parse(text)
    assert len(jd.required_skills) == 0

# ---------------------------------------------------------------------------
# RESPONSIBILITIES
# ---------------------------------------------------------------------------
def test_responsibilities_section_lines_extracted():
    text = """
    Responsibilities
    - Develop scalable applications.
    - Maintain existing infrastructure.
    """
    jd = _parse(text)
    assert len(jd.responsibilities) == 2
    assert "Develop scalable applications." in jd.responsibilities[0]

def test_responsibilities_requirement_line_not_responsibility():
    text = """
    Requirements
    - Must have 3 years of experience.
    """
    jd = _parse(text)
    assert len(jd.responsibilities) == 0

# ---------------------------------------------------------------------------
# DELIVERABLES
# ---------------------------------------------------------------------------
def test_deliverables_develop_ml_model():
    text = """
    Responsibilities
    - Develop ML models for fraud detection.
    """
    jd = _parse(text)
    assert len(jd.key_deliverables) == 1
    assert "ML models" in jd.key_deliverables[0]

def test_deliverables_collaborate_with_team():
    text = """
    Responsibilities
    - Collaborate with team members.
    """
    jd = _parse(text)
    assert len(jd.key_deliverables) == 0

# ---------------------------------------------------------------------------
# CERTIFICATIONS
# ---------------------------------------------------------------------------
def test_certifications_toeic_accepted():
    text = """
    Requirements
    - TOEIC 650
    """
    jd = _parse(text)
    assert any("TOEIC" in c for c in jd.required_certifications)

def test_certifications_aws_alone_not_certification():
    text = """
    Requirements
    - AWS and Azure experience
    """
    jd = _parse(text)
    assert len(jd.required_certifications) == 0

def test_certifications_aws_certified():
    text = """
    Requirements
    - AWS Certified Solutions Architect
    """
    jd = _parse(text)
    assert any("AWS Certified" in c for c in jd.required_certifications)

# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
def test_title_heading_is_not_title():
    text = """
    # REQUIREMENTS
    - Python
    """
    jd = _parse(text, "Fallback Title")
    assert jd.job_title == "Fallback Title"

def test_title_ai_engineer_intern_preserved():
    text = """
    AI Engineer Intern
    - Python
    """
    jd = _parse(text)
    assert jd.job_title == "AI Engineer Intern"

# ---------------------------------------------------------------------------
# COMPANY
# ---------------------------------------------------------------------------
def test_company_explicit_extract():
    text = """
    AI Engineer Intern
    Company: Viettel
    """
    jd = _parse(text)
    assert jd.company_name == "Viettel"

def test_company_no_explicit_company():
    text = """
    AI Engineer Intern
    We are a fast-growing startup.
    """
    jd = _parse(text)
    assert jd.company_name is None

# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
def test_overview_section_used():
    text = """
    Job Overview
    We are looking for a great developer.
    
    Requirements
    - Python
    """
    jd = _parse(text)
    assert "looking for a great developer" in jd.job_overview
    assert "Python" not in jd.job_overview

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def test_output_lists_deduped_order_preserving():
    text = """
    Requirements
    - Python
    - Git
    - Python
    """
    jd = _parse(text)
    assert jd.required_skills == ["Python", "Git"]
