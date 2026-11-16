"""Presentation-only adapters; canonical parsed objects are never mutated."""
from __future__ import annotations
from copy import deepcopy
from typing import Any
from src.Normalization.cv_normalizer import normalize_cv
from src.Normalization.skill_normalizer import normalize_skill
from src.Representation.feature_builder import build_cv_features, CVFeatures
from src.Normalization.jd_normalizer import _atomic_fields

CV_CATEGORIES = ["Education", "Skills", "Experience", "Projects"]
JD_CATEGORIES = ["Education", "Required Skills", "Preferred Skills", "Responsibilities", "Certifications"]

def _data(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(record, dict) or not isinstance(record.get("parsed_data"), dict):
        raise ValueError("expected parsed wrapper with parsed_data object")
    return str(record.get("id", "document")), deepcopy(record["parsed_data"])

def _item(document_id, index, name, category, path, source_type, **extra):
    return {"id": f"{document_id}_{category.lower().replace(' ', '_')}_{index:03d}", "name": str(name), "category": category, "source_path": path, "source_type": source_type, **extra}

def parsed_cv_to_ui_features(parsed_cv: dict[str, Any]) -> list[dict[str, Any]]:
    """Project a parsed CV through normalization and FeatureBuilder.

    The public name is retained for ConfirmOverride compatibility; callers
    still pass the immutable parsed wrapper, while presentation is built from
    the deterministic normalized/typed view.
    """
    if not isinstance(parsed_cv, dict) or not isinstance(parsed_cv.get("parsed_data"), dict):
        raise ValueError("expected parsed wrapper with parsed_data object")
    return cv_features_to_ui_features(build_cv_features(normalize_cv(parsed_cv)))


def cv_features_to_ui_features(features: CVFeatures) -> list[dict[str, Any]]:
    doc = str(features.id or "document"); out=[]
    d = features.education
    degree=d.get("education_degree"); field=d.get("education_field")
    if degree is None: degree = d.get("degree")
    if field is None: field = d.get("field")
    if degree: out.append(_item(doc,len(out)+1,f"Degree: {degree}","Education","parsed_data.education_degree","education_degree"))
    if field: out.append(_item(doc,len(out)+1,f"Field: {field}","Education","parsed_data.education_field","education_field"))
    for i, history in enumerate(d.get("history") or [],1):
        if isinstance(history, dict):
            label = " — ".join(str(history.get(k)) for k in ("degree", "field_of_study", "institution") if history.get(k))
            if label:
                edu_index = sum(1 for x in out if x["category"] == "Education") + 1
                out.append(_item(doc,edu_index,label,"Education",f"parsed_data.education_history[{i-1}]","education_history"))
    for i, skill in enumerate(features.skill_provenance.get("explicit", []) or [],1):
        if isinstance(skill,str) and skill.strip(): out.append(_item(doc,i,skill.strip(),"Skills",f"parsed_data.skills[{i-1}]","explicit_skill"))
    seen={x["name"].casefold(): x for x in out if x["category"]=="Skills"}
    project_paths = {}
    for pi, project in enumerate(features.projects or [], 1):
        if isinstance(project, dict):
            for ti, tech in enumerate(project.get("technologies") or [], 1):
                if isinstance(tech, str) and tech.strip():
                    project_paths.setdefault(normalize_skill(tech).casefold(), f"parsed_data.projects[{pi-1}].technologies[{ti-1}]")
    for i, skill in enumerate(features.skill_provenance.get("project_derived", []) or [],1):
        if not isinstance(skill,str) or not skill.strip(): continue
        key=skill.casefold()
        if key in seen:
            path = project_paths.get(key)
            if path:
                seen[key].setdefault("source_paths", [seen[key]["source_path"]]).append(path)
            continue
        item=_item(doc,len(seen)+1,skill.strip(),"Skills",project_paths.get(key, f"normalized.skills.project_derived[{i-1}]"),"project_technology")
        seen[key]=item; out.append(item)
    for i, project in enumerate(features.projects or [],1):
        if not isinstance(project,dict): continue
        name=project.get("name")
        if name: out.append(_item(doc,i,str(name),"Projects",f"parsed_data.projects[{i-1}].name","project"))
    years=features.professional_years
    if years is not None: out.append(_item(doc,1,f"Experience: {years} years","Experience","parsed_data.experience_years","experience_years"))
    for i, work in enumerate(features.work_evidence or [],1):
        if not isinstance(work,dict): continue
        role=work.get("role")
        if role:
            exp_index = sum(1 for x in out if x["category"] == "Experience") + 1
            out.append(_item(doc,exp_index,str(role),"Experience",f"parsed_data.work_experience[{i-1}].role","job_role"))
        for j, resp in enumerate(work.get("responsibilities_and_impact") or [],1):
            if isinstance(resp,str) and resp.strip():
                exp_index = sum(1 for x in out if x["category"] == "Experience") + 1
                out.append(_item(doc,exp_index,resp.strip(),"Experience",f"parsed_data.work_experience[{i-1}].responsibilities_and_impact[{j-1}]","responsibility"))
    return out

def parsed_jd_to_ui_features(parsed_jd: dict[str, Any]) -> list[dict[str, Any]]:
    doc,d=_data(parsed_jd); out=[]
    if d.get("required_degree"): out.append(_item(doc,1,f"Minimum degree: {d['required_degree']}","Education","parsed_data.required_degree","required_degree"))
    for i,v in enumerate(_atomic_fields(d.get("preferred_fields") or []),1):
        if isinstance(v,str) and v.strip():
            edu_index = sum(1 for x in out if x["category"] == "Education") + 1
            out.append(_item(doc,edu_index,f"Preferred field: {v.strip()}","Education",f"parsed_data.preferred_fields[{i-1}]","preferred_field"))
    for category,key,source in (("Required Skills","required_skills","required_skill"),("Preferred Skills","preferred_skills","preferred_skill"),("Responsibilities","responsibilities","responsibility"),("Certifications","required_certifications","certification")):
        for i,v in enumerate(d.get(key) or [],1):
            if isinstance(v,str) and v.strip(): out.append(_item(doc,i,v.strip(),category,f"parsed_data.{key}[{i-1}]",source))
    return out

def get_cv_ui_categories(): return list(CV_CATEGORIES)
def get_jd_ui_categories(): return list(JD_CATEGORIES)
