import pytest
from app.api.services.runtime_explanation_service import _validate_narrative

def pre(maxq=3):
    return {'facts':{'strengths':[{'fact_id':'str_001','evidence_refs':['ev_skill']}], 'required_skills_no_evidence':['Git'], 'weak_experience_evidence':[{'fact_id':'gap_001','evidence_refs':['ev_exp']}]}, 'interview_topics':[{'fact_id':'int_001','evidence_refs':['ev_exp']}], 'interview_config':{'max_questions':maxq}, 'selected_evidence':{'ev_skill':{},'ev_exp':{}}}
def test_cross_fact_reference_rejected():
    with pytest.raises(ValueError): _validate_narrative({'summary':'x','strengths':[{'fact_id':'str_001','text':'x','evidence_refs':['ev_exp']}],'gaps':[],'interview_focus':[]},pre())
def test_dynamic_question_limit():
    raw={'summary':'x','strengths':[],'gaps':[],'interview_focus':[{'fact_id':'int_001','topic':'x','question':'q','reason':'r','evidence_refs':['ev_exp']}]*3}
    with pytest.raises(ValueError): _validate_narrative(raw,pre(1))
def test_missing_skill_has_empty_refs_and_unknown_fact_rejected():
    with pytest.raises(ValueError): _validate_narrative({'summary':'x','strengths':[],'gaps':[{'fact_id':'gap_missing_001','type':'required_skill_no_evidence','text':'Git','evidence_refs':['ev_skill']}],'interview_focus':[]},pre())
