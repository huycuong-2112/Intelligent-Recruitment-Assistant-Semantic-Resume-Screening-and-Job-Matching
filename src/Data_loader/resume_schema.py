from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ResumeSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: str = Field(description="Unique resume identifier")
    filename: str = Field(description="Original resume file name")

    text: str = Field(
        default="",
        description="Raw text extracted from the resume",
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Normalized skills extracted from the resume",
    )

    experience_years: Optional[float] = Field(
        default=None,
        description="Estimated years of experience",
    )

    education: Optional[str] = Field(
        default=None,
        description="Highest education level or summary",
    )