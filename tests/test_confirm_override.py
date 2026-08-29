from copy import deepcopy
import pytest
from app.api.adapters.confirm_override import build_confirm_override, apply_cv_override, apply_jd_override
from app.api.adapters.presentation_adapter import parsed_cv_to_ui_features, parsed_jd_to_ui_features
from src.Normalization.cv_normalizer import normalize_cv
from src.Normalization.jd_normalizer import normalize_jd

def cv(): return {"id":"cv_001","parsed_data":{"skills":["Python","Git"],"projects":[{"name":"Demo","technologies":["Python","Docker"]}],"experience_years":2.0,"education_degree":"Bachelor"}}
def jd(): return {"id":"jd_001","parsed_data":{"required_skills":["Python","Git"],"preferred_skills":["Docker"],"responsibilities":["Build ML models"],"required_degree":"Bachelor"}}

def test_cv_roundtrip_removes_all_python_sources_and_preserves_canonical():
    canonical=cv(); features=parsed_cv_to_ui_features(canonical); before=deepcopy(canonical)
    kept=[f for f in features if f["name"]!="Python"]; ov=build_confirm_override("cv_001",features,kept)
    result=apply_cv_override(canonical,ov); runtime=result["runtime_document"]
    assert canonical==before and "Python" not in normalize_cv(runtime)["skills"]["all"]
    assert "Git" in normalize_cv(runtime)["skills"]["all"] and "Docker" in normalize_cv(runtime)["skills"]["all"]
    assert "Python" in normalize_cv(canonical)["skills"]["all"]

def test_cv_manual_skill_and_unsupported_education():
    c=cv(); fs=parsed_cv_to_ui_features(c); manual={"name":"Kubernetes","category":"Skills","source_type":"manual_ui"}; ov=build_confirm_override("cv_001",fs,fs+[manual]); r=apply_cv_override(c,ov); assert "Kubernetes" in r["runtime_document"]["parsed_data"]["skills"]
    bad=build_confirm_override("cv_001",fs,fs+[{"name":"MSc","category":"Education","source_type":"manual_ui"}]); assert r["runtime_document"]
    assert apply_cv_override(c,bad)["unsupported_actions"]

def test_jd_required_preferred_and_responsibility_targets():
    c=jd(); fs=parsed_jd_to_ui_features(c); kept=[f for f in fs if f["name"]!="Git"]; kept += [{"name":"PyTorch","category":"Preferred Skills","source_type":"manual_ui"},{"name":"Evaluate datasets","category":"Responsibilities","source_type":"manual_ui"}]
    r=apply_jd_override(c,build_confirm_override("jd_001",fs,kept)); d=r["runtime_document"]["parsed_data"]; assert d["required_skills"]==["Python"]; assert d["preferred_skills"]==["Docker","PyTorch"]; assert d["responsibilities"]==["Build ML models","Evaluate datasets"]; assert normalize_jd(r["runtime_document"])["skills"]["required"]==["Python"]

def test_safety_identity_determinism_and_idempotence():
    c=cv(); fs=parsed_cv_to_ui_features(c); ov=build_confirm_override("cv_001",fs,fs); assert ov==build_confirm_override("cv_001",fs,fs); assert apply_cv_override(c,ov)==apply_cv_override(c,ov)
    with pytest.raises(ValueError): apply_cv_override(c,{"document_id":"cv_999","removed_features":[]})
    with pytest.raises(ValueError): apply_cv_override(c,{"document_id":"cv_001","removed_features":[{"id":"x","source_path":"__class__"}]})

def test_scalar_removal_and_malformed_override():
    c=cv(); f=parsed_cv_to_ui_features(c); kept=[x for x in f if x.get("source_type")!="experience_years"]; ov=build_confirm_override("cv_001",f,kept); assert apply_cv_override(c,ov)["runtime_document"]["parsed_data"]["experience_years"] is None
    with pytest.raises(ValueError): build_confirm_override("",f,[])
