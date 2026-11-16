from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DegreeType = Literal["High School", "Associate", "Bachelor", "Engineer", "Master", "Ph.D", "Other"]


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMA DEFINITIONS
# ---------------------------------------------------------------------------
class EducationItem(BaseModel):
    institution: str = Field(..., description="Tên trường đại học / cao đẳng / viện đào tạo")
    degree: Optional[DegreeType] = Field(None, description="Bằng cấp cao nhất đạt được tại trường")
    field_of_study: Optional[str] = Field(None, description="Chuyên ngành đào tạo")
    start_year: Optional[int] = Field(None, description="Năm bắt đầu học")
    end_year: Optional[int] = Field(None, description="Năm tốt nghiệp (hoặc dự kiến tốt nghiệp)")
    gpa_or_honors: Optional[str] = Field(None, description="Điểm GPA hoặc học bổng/danh hiệu đạt được")


class ExperienceItem(BaseModel):
    company: str = Field(..., description="Tên công ty / tổ chức / doanh nghiệp làm việc")
    role: str = Field(..., description="Chức danh / vị trí công việc")
    start_date: Optional[str] = Field(None, description="Thời gian bắt đầu (VD: '10/2023', '2022')")
    end_date: Optional[str] = Field(None, description="Thời gian kết thúc (VD: '05/2024', 'Present')")
    responsibilities_and_impact: List[str] = Field(
        default_factory=list,
        description="Danh sách các đầu việc, trách nhiệm và thành tựu đã thực hiện"
    )


class ProjectItem(BaseModel):
    name: str = Field(..., description="Tên dự án hoặc đề tài kỹ thuật")
    role: Optional[str] = Field(None, description="Vai trò trong dự án (VD: Leader, Developer, Researcher)")
    technologies: List[str] = Field(default_factory=list, description="Công cụ, ngôn ngữ, framework sử dụng")
    description: str = Field(..., description="Mô tả bài toán kỹ thuật, giải pháp và tính năng")
    impact_metrics: List[str] = Field(
        default_factory=list,
        description="Các chỉ số định lượng đo lường hiệu năng (VD: 'Tăng 30% doanh thu', '45 FPS')"
    )
    links: List[str] = Field(default_factory=list, description="Link demo, source code GitHub hoặc Paper")
    heuristic_score: Optional[float] = Field(
        None,
        description="Điểm đánh giá sơ bộ độ phức tạp và tính định lượng của dự án (0.0 - 1.0)"
    )


class StructuredResume(BaseModel):
    summary: Optional[str] = Field(None, description="Tóm tắt hồ sơ năng lực / Mục tiêu nghề nghiệp")
    education_degree: Optional[DegreeType] = Field(None, description="Bằng cấp cao nhất của ứng viên")
    education_field: Optional[str] = Field(None, description="Chuyên ngành đào tạo chính")
    education_history: List[EducationItem] = Field(default_factory=list, description="Lịch sử học vấn")
    experience_years: float = Field(
        0.0,
        ge=0.0,
        le=50.0,
        description="Tổng số năm kinh nghiệm làm việc tích lũy (làm tròn 1 chữ số thập phân). Mốc hiện tại tính đến 2026."
    )
    job_titles: List[str] = Field(default_factory=list, description="Các chức danh công việc đã đảm nhiệm")
    work_experience: List[ExperienceItem] = Field(
        default_factory=list,
        description="Danh sách toàn bộ các công ty và vị trí công việc từng làm"
    )
    projects: List[ProjectItem] = Field(
        default_factory=list,
        description="Danh sách các dự án tiêu biểu, đồ án hoặc nghiên cứu"
    )
    skills: List[str] = Field(default_factory=list, description="Toàn bộ kỹ năng chuyên môn kỹ thuật và mềm")
    certifications: List[str] = Field(default_factory=list, description="Chứng chỉ chuyên môn hoặc giấy phép hành nghề")

    @field_validator("skills", "job_titles", "certifications", mode="before")
    @classmethod
    def deduplicate_string_list(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            if isinstance(item, str):
                s = item.strip().strip("•-* \t")
                if s and s not in cleaned:
                    cleaned.append(s)
        return cleaned

    @field_validator("experience_years", mode="before")
    @classmethod
    def clean_exp_years(cls, value: Any) -> float:
        try:
            return max(0.0, round(float(value), 1))
        except (ValueError, TypeError):
            return 0.0
