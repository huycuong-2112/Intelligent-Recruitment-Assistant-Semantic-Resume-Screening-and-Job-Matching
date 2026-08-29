import json
from pathlib import Path
import pytest
import yaml

from app.api.services.runtime_matching_config import load_runtime_matching_config, load_runtime_mdms_weights, RuntimeMatchingConfigError

ROOT = Path(__file__).resolve().parents[1]

def test_it_runtime_config_and_provenance():
    cfg = load_runtime_matching_config("IT", ROOT)
    assert cfg["weights"] == {"skill": .4, "experience": .2, "education": .1, "semantic": .3}
    assert set(cfg["weights"]) == {"skill", "experience", "education", "semantic"}
    assert sum(cfg["weights"].values()) == pytest.approx(1.0)
    assert cfg["selection_scope"] == "development" and cfg["blind_evaluated"] is False

def test_validation_no_fallback(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/mdms.yaml").read_text())
    cfg["mdms"].pop("runtime_weights")
    (tmp_path / "configs").mkdir(); (tmp_path / "configs/mdms.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeMatchingConfigError): load_runtime_mdms_weights("IT", tmp_path)

@pytest.mark.parametrize("mutator", [
    lambda w: w.update(skill=-.1),
    lambda w: w.pop("semantic"),
    lambda w: w.update(skill="bad"),
    lambda w: w.update(skill=.9),
])
def test_malformed_weights_rejected(tmp_path, mutator):
    cfg = yaml.safe_load((ROOT / "configs/mdms.yaml").read_text()); mutator(cfg["mdms"]["runtime_weights"])
    (tmp_path / "configs").mkdir(); (tmp_path / "configs/mdms.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeMatchingConfigError): load_runtime_mdms_weights("IT", tmp_path)

def test_domain_required_and_unrelated_rejected():
    with pytest.raises(RuntimeMatchingConfigError): load_runtime_mdms_weights("", ROOT)
    with pytest.raises(RuntimeMatchingConfigError): load_runtime_mdms_weights("Finance", ROOT)

def test_component_matcher_parameters_unchanged():
    cfg = yaml.safe_load((ROOT / "configs/mdms.yaml").read_text())
    assert cfg["matching"]["skills"]["required_weight"] == 1.0
    assert cfg["matching"]["experience"]["years_weight"] == .3
    assert cfg["matching"]["experience"]["evidence_weight"] == .7
    assert cfg["matching"]["education"]["degree_weight"] == .6
    assert cfg["matching"]["education"]["field_weight"] == .4
