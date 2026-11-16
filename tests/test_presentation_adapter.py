import copy
import pytest
from app.api.adapters.presentation_adapter import parsed_cv_to_ui_features, parsed_jd_to_ui_features

def cv(): return {"id":"cv_001","parsed_data":{"education_degree":"Bachelor","education_field":"Computer Science","skills":["Python","Git"],"experience_years":2.0,"work_experience":[{"role":"Engineer","responsibilities_and_impact":["Built systems"]}],"projects":[{"name":"Demo","technologies":["Python","Docker"],"description":"x"}]}}
def jd(): return {"id":"jd_001","parsed_data":{"required_degree":"Bachelor","preferred_fields":["Data Science"],"required_skills":["Python"],"preferred_skills":["Docker"],"responsibilities":["Build models"],"required_certifications":[]}}
def test_cv_deterministic_and_provenance():
 x=cv(); before=copy.deepcopy(x); a=parsed_cv_to_ui_features(x); assert x==before and a==parsed_cv_to_ui_features(x); assert all(set(['id','name','category','source_path','source_type'])<=set(i) for i in a); assert [i for i in a if i['name']=='Python'][0]['source_type']=='explicit_skill'
def test_jd_required_preferred_and_responsibility_distinct():
 a=parsed_jd_to_ui_features(jd()); assert any(i['category']=='Required Skills' and i['source_type']=='required_skill' for i in a); assert any(i['category']=='Preferred Skills' and i['source_type']=='preferred_skill' for i in a); assert any(i['category']=='Responsibilities' for i in a); assert not any(i['category']=='Domain' for i in a)
def test_malformed_fails_and_empty_safe():
 with pytest.raises(ValueError): parsed_cv_to_ui_features({'id':'x'})
 assert parsed_cv_to_ui_features({'id':'x','parsed_data':{}})==[]

def test_skill_features_only_come_from_approved_skill_sources():
    assert [x for x in parsed_cv_to_ui_features({'id':'x','parsed_data':{'skills':[],'projects':[]}}) if x['category']=='Skills'] == []
    skills = parsed_cv_to_ui_features({'id':'x','parsed_data':{'skills':['Python'],'projects':[]}})
    assert len([x for x in skills if x['category']=='Skills']) == 1
    projects = parsed_cv_to_ui_features({'id':'x','parsed_data':{'skills':[],'projects':[{'name':'Demo','technologies':['Docker']}]}})
    docker = [x for x in projects if x['category']=='Skills'][0]
    assert docker['source_type'] == 'project_technology'
