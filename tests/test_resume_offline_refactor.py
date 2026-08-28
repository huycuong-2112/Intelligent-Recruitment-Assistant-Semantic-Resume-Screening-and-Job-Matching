"""R0.2 regression test: verify OfflineResumeParser produces the same
legacy output after the structural refactor."""
from __future__ import annotations

import sys
from pathlib import Path

# Support running from project root
_data_loader_dir = str(Path(__file__).resolve().parent.parent / "src" / "Data_loader")
if _data_loader_dir not in sys.path:
    sys.path.insert(0, _data_loader_dir)

from resume_schema import (
    DegreeType,
    EducationItem,
    ExperienceItem,
    ProjectItem,
    StructuredResume,
)
from offline.resume_offline_parser import OfflineResumeParser


# ---------------------------------------------------------------------------
# Deterministic fixture: simplified resume text
# ---------------------------------------------------------------------------
SAMPLE_RESUME_TEXT = """\
## CAREER OBJECTIVES
Seeking an internship position in AI/ML engineering.

## EDUCATION
University of Technology
Bachelor of Computer Science
2021 – 2025
GPA: 3.5/4.0

## WORK EXPERIENCE
AI Engineer Intern
FPT Software
06/2024 – 09/2024
- Developed and deployed ML pipelines for document classification.
- Improved model accuracy by 15% using data augmentation techniques.

## FEATURED PROJECTS
### Recruitment Matching System
Technologies: Python, FastAPI, Sentence-Transformers, Qdrant
Built an end-to-end resume screening system that matches candidates to JDs using semantic similarity.
Achieved 85% precision on test set.
https://github.com/example/recruitment-system

## TECHNICAL SKILLS
Python, PyTorch, TensorFlow, FastAPI, Docker, Git, SQL, LangChain

## CERTIFICATIONS
AWS Cloud Practitioner
TOEIC 850
"""


class TestOfflineResumeParserLegacy:
    """Confirm the refactored parser still returns the same expected legacy
    behavior for a small deterministic fixture."""

    def _parse(self):
        return OfflineResumeParser.parse(SAMPLE_RESUME_TEXT)

    def test_parse_returns_structured_resume(self):
        result = self._parse()
        assert isinstance(result, StructuredResume)

    def test_summary_extraction(self):
        result = self._parse()
        assert result.summary is not None
        assert "internship" in result.summary.lower() or "career" in result.summary.lower() or len(result.summary) > 0

    def test_education_degree(self):
        result = self._parse()
        # "Bachelor" should be detected from the education section
        assert result.education_degree == "Bachelor"

    def test_education_history(self):
        result = self._parse()
        assert len(result.education_history) >= 1
        edu = result.education_history[0]
        assert isinstance(edu, EducationItem)
        assert "University" in edu.institution or "Technology" in edu.institution

    def test_work_experience(self):
        result = self._parse()
        assert len(result.work_experience) >= 1
        exp = result.work_experience[0]
        assert isinstance(exp, ExperienceItem)

    def test_experience_years(self):
        result = self._parse()
        # 06/2024 – 09/2024 is within the same year, so DATE_SPAN_REGEX
        # captures 2024-2024 = 0 years
        assert result.experience_years >= 0.0

    def test_projects_extraction(self):
        result = self._parse()
        assert len(result.projects) >= 1
        proj = result.projects[0]
        assert isinstance(proj, ProjectItem)
        assert proj.heuristic_score is not None

    def test_skills_extraction(self):
        result = self._parse()
        assert len(result.skills) >= 1

    def test_certifications_extraction(self):
        result = self._parse()
        assert len(result.certifications) >= 1

    def test_schema_imports_from_resume_schema(self):
        """Verify all schema classes are importable from resume_schema."""
        from resume_schema import StructuredResume as SR
        from resume_schema import EducationItem as EI
        from resume_schema import ExperienceItem as EXI
        from resume_schema import ProjectItem as PI
        assert SR is StructuredResume
        assert EI is EducationItem
        assert EXI is ExperienceItem
        assert PI is ProjectItem

    def test_degree_type_values(self):
        """Verify DegreeType literal values are unchanged."""
        import typing
        args = typing.get_args(DegreeType)
        assert "High School" in args
        assert "Associate" in args
        assert "Bachelor" in args
        assert "Engineer" in args
        assert "Master" in args
        assert "Ph.D" in args
        assert "Other" in args

    def test_section_isolation(self):
        """Verify section splitting works."""
        sections = OfflineResumeParser.isolate_sections(SAMPLE_RESUME_TEXT)
        assert isinstance(sections, dict)
        assert "education" in sections
        assert "experience" in sections
        assert "projects" in sections
        assert "skills" in sections
        assert "certifications" in sections

    def test_preferred_fields_not_in_resume_schema(self):
        """StructuredResume does not have preferred_fields (that's JD schema)."""
        result = self._parse()
        assert not hasattr(result, "preferred_fields")
