import json
from pathlib import Path
import pytest
from src.Explainability.pre_explanation_builder import build_pre_explanation

ROOT=Path(__file__).resolve().parents[1]
def load(): return json.loads((ROOT/'tests/fixtures/xai/cv_001_xai.json').read_text(encoding='utf-8'))
def test_projection_is_deterministic_and_preserves_decision():
    x=load(); a=build_pre_explanation(x); b=build_pre_explanation(x)
    assert a==b and a['decision']['final_score']==x['decision']['final_score'] and a['decision']['coverage']==x['decision']['coverage']
    assert set(a['decision']['dimensions'])=={'skill','experience','education','semantic'}
    assert a['decision']['weights']==x['decision']['weights'] and abs(sum(a['decision']['weights'].values())-1)<1e-9
def test_missing_required_and_weak_experience_are_compact():
    a=build_pre_explanation(load()); assert 'Python programming' in a['facts']['required_skills_no_evidence']; assert len(a['facts']['weak_experience_evidence'])<=2; assert not any('preferred' in str(g).lower() for g in a['facts']['required_skills_no_evidence']); assert any(t['reason']=='weak_experience_evidence' for t in a['interview_topics']); assert len(a['interview_topics'])<=3
def test_refs_resolve_and_unknown_is_not_gap():
    a=build_pre_explanation(load()); refs={r for section in [a['facts']['strengths'],a['facts']['weak_experience_evidence'],a['interview_topics']] for i in section for r in i.get('evidence_refs',[])}; assert refs<=set(a['selected_evidence']); assert a['facts']['unknowns']==[]
def test_unresolved_reference_fails():
    x=load(); x['strength_candidates']=[{'type':'x','evidence_ref':'missing'}]
    with pytest.raises(ValueError): build_pre_explanation(x)
