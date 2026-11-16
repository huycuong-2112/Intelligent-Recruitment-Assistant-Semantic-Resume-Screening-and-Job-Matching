"""Category-safe manual feature types shared by the CV Refine UI."""

TYPE_OPTIONS = {
    "Skills": [("skill", "Skill")],
    "Education": [("degree", "Degree"), ("field", "Field of Study")],
    "Experience": [("role", "Role / Job Title"), ("responsibility", "Responsibility / Evidence")],
    "Projects": [("project_name", "Project Name"), ("project_evidence", "Project Evidence")],
}


def types_for(category: str) -> list[str]:
    return [value for value, _ in TYPE_OPTIONS.get(category, [])]


def labels_for(category: str) -> list[str]:
    return [label for _, label in TYPE_OPTIONS.get(category, [])]


def subtype_selector_visible(category: str) -> bool:
    return len(TYPE_OPTIONS.get(category, [])) > 1
