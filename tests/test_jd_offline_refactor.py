"""R0.1 regression test: verify OfflineJDExtractor produces the same
legacy output after the structural refactor."""
from __future__ import annotations

import sys
from pathlib import Path

# Support running from project root
_data_loader_dir = str(Path(__file__).resolve().parent.parent / "src" / "Data_loader")
if _data_loader_dir not in sys.path:
    sys.path.insert(0, _data_loader_dir)

from jd_schema import StructuredJobDescription
from offline.jd_offline_parser import OfflineJDExtractor


# ---------------------------------------------------------------------------
# Deterministic fixture: simplified AI Engineer Intern JD
# ---------------------------------------------------------------------------
SAMPLE_JD_TEXT = """\
## AI Engineer Intern
We are looking for a passionate AI Engineer Intern to join our team.
1 year of experience required in Python or ML frameworks.
Research & Implement: Participate in researching and implementing algorithms for data science and AI projects.
Cloud: Basic awareness of cloud platforms (AWS, Azure, or GCP) is an advantage.
"""

EXPECTED_TITLE = "AI Engineer Intern"
EXPECTED_EXPERIENCE = 1.0
EXPECTED_DEGREE = "Engineer"


class TestOfflineJDExtractorLegacy:
    """Confirm the refactored parser still returns the same expected legacy
    behavior for a small deterministic fixture."""

    def test_parse_returns_structured_jd(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert isinstance(result, StructuredJobDescription)

    def test_title_extraction(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert result.job_title == EXPECTED_TITLE

    def test_experience_extraction(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert result.min_experience_years == EXPECTED_EXPERIENCE

    def test_degree_extraction_legacy_engineer_bug(self):
        """The current parser returns 'Engineer' because 'engineer' appears
        in the title before 'bachelor'. This is known legacy behavior."""
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert result.required_degree == EXPECTED_DEGREE

    def test_cloud_awareness_is_not_misclassified_as_certification(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert not any("AWS" in c or "Azure" in c or "GCP" in c for c in result.required_certifications)

    def test_responsibilities_extraction(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert len(result.responsibilities) >= 1

    def test_empty_preferred_skills(self):
        """Legacy parser does not classify preferred vs required skills."""
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert result.preferred_skills == []

    def test_preferred_fields_only_from_preferred_section(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert result.preferred_fields == []

    def test_company_name_is_none(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert result.company_name is None

    def test_job_overview_is_text_prefix(self):
        result = OfflineJDExtractor.parse(SAMPLE_JD_TEXT, "fallback_title")
        assert result.job_overview == SAMPLE_JD_TEXT[:350].strip()

    def test_schema_import_from_jd_schema(self):
        """Verify StructuredJobDescription is importable from jd_schema."""
        from jd_schema import StructuredJobDescription as SJD
        assert SJD is StructuredJobDescription

    def test_degree_type_values(self):
        """Verify DegreeType literal values are unchanged."""
        from jd_schema import DegreeType
        import typing
        args = typing.get_args(DegreeType)
        assert "High School" in args
        assert "Associate" in args
        assert "Bachelor" in args
        assert "Engineer" in args
        assert "Master" in args
        assert "Ph.D" in args
        assert "Any" in args
        assert "Other" in args
