from src.Matching.education_matcher import match_education

def test_degree_field_and_not_applicable():
    jd = {"education": {"minimum_degree": "Bachelor", "preferred_fields": ["Computer Science"]}}
    good = match_education(jd, {"education": {"degree": "Bachelor", "field": "Computer Science"}})
    low = match_education(jd, {"education": {"degree": "Associate", "field": "Computer Science"}})
    unknown = match_education(jd, {"education": {"degree": None, "field": None}})
    assert good["score"] == 1.0 and low["degree"]["score"] == 0.0 and unknown["degree"]["score"] is None
    assert match_education({"education": {}}, {"education": {}})["degree"]["availability"] == "not_applicable"

def test_evaluable_absence_is_no_evidence_and_unavailable_is_unknown():
    jd={"education":{"minimum_degree":"Bachelor","preferred_fields":["Computer Science"]}}
    evaluable={"skills":{"all":["Python"]},"experience":{"work_evidence":["built systems"]},"education":{"degree":None,"field":None}}
    result=match_education(jd,evaluable)
    assert result["degree"]["score"] == 0.0 and result["degree"]["status"] == "no_evidence"
    unavailable={"source_status":"FAILED","education":{"degree":None,"field":None}}
    assert match_education(jd,unavailable)["degree"]["score"] is None
