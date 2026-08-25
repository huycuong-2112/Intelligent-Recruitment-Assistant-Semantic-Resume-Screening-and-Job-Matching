from src.Baselines.rule_based_baseline import run_rule_based

def test_rule_based_transparent_components():
    jd={"id":"j","skills":{"required":["Python"],"preferred":["Git"]},"experience":{"minimum_years":1},"education":{}}
    cv={"id":"c","skills":{"all":["Python"]},"experience":{"professional_years":0.0},"education":{}}
    result=run_rule_based(jd,cv)
    assert result["components"]["skill"] > 0 and result["components"]["experience"] == 0.0
    assert result["weights"]["weight_policy"] == "HEURISTIC_NOT_OPTIMIZED"
    unknown=run_rule_based(jd,{"id":"c","skills":{"all":[]},"experience":{"professional_years":None},"education":{}})
    assert unknown["components"]["experience"] is None
