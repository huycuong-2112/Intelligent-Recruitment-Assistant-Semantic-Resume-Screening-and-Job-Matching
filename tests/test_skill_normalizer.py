from src.Normalization.skill_normalizer import normalize_skill, normalize_skills


def test_aliases_and_deduplication():
    assert [normalize_skill(x) for x in ["RESTful APIs", "Git/GitHub", "Pho BERT", "LLMs", "Auto Gen"]] == ["REST API", "Git", "PhoBERT", "LLM", "AutoGen"]
    assert normalize_skills(["Git", "git/github", "FastAPI"]) == ["Git", "FastAPI"]
