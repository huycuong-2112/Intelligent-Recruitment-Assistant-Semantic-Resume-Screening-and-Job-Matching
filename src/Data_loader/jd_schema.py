from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DegreeType = Literal["High School", "Associate", "Bachelor", "Engineer", "Master", "Ph.D", "Any", "Other"]


# ---------------------------------------------------------------------------
# 1-TO-1 JD PYDANTIC SCHEMA
# ---------------------------------------------------------------------------
class StructuredJobDescription(BaseModel):
    job_title: str = Field(..., description="Chức danh công việc cần tuyển dụng")
    company_name: Optional[str] = Field(None, description="Tên công ty / doanh nghiệp tuyển dụng")
    job_overview: Optional[str] = Field(None, description="Mô tả tóm tắt vai trò và sứ mệnh của vị trí")
    min_experience_years: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="Số năm kinh nghiệm tối thiểu yêu cầu (VD: ghi '2+ năm' lấy 2.0; không yêu cầu lấy 0.0)"
    )
    required_degree: Optional[DegreeType] = Field(
        default=None,
        description="Bằng cấp tối thiểu: 'Associate', 'Bachelor', 'Engineer', 'Master', 'Ph.D', 'Any'"
    )
    preferred_fields: List[str] = Field(
        default_factory=list,
        description="Danh sách các chuyên ngành đào tạo ưu tiên (VD: ['Computer Science', 'Mechatronics'])"
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Kỹ năng kỹ thuật và nghiệp vụ BẮT BUỘC (Must-have)"
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Kỹ năng ưu tiên / điểm cộng (Nice-to-have)"
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Danh sách các đầu việc và trách nhiệm chính của vị trí"
    )
    key_deliverables: List[str] = Field(
        default_factory=list,
        description="Các dự án hoặc bài toán cốt lõi cần giải quyết"
    )
    required_certifications: List[str] = Field(
        default_factory=list,
        description="Chứng chỉ bắt buộc hoặc ưu tiên (TOEIC, IELTS, CFA, AWS, PMP...)"
    )

    @field_validator("required_skills", "preferred_skills", "preferred_fields", "required_certifications", mode="before")
    @classmethod
    def deduplicate_list(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if isinstance(item, str):
                s = item.strip().strip("•-* \t\n")
                if s and s not in cleaned:
                    cleaned.append(s)
        return cleaned

    @field_validator("min_experience_years", mode="before")
    @classmethod
    def clean_exp_years(cls, value: Any) -> float:
        try:
            return max(0.0, round(float(value), 1))
        except (ValueError, TypeError):
            return 0.0
