import json, pickle
from pathlib import Path
import pytest

from app.api.core.config import settings
from app.api.services.runtime_preparation_service import prepare_cv_runtime, prepare_jd_runtime, RuntimePreparationError

RUN = "a" * 32

class FakeEmb:
    model_name = "fake-test-model"
    dimension = 3
    normalize_embeddings = True
    def embed_batch(self, texts): return [[float(i), 1.0, 0.0] for i, _ in enumerate(texts)]
    def embed_text(self, text): return [1.0, 0.0, 0.0]

def setup_runtime(tmp_path, kind, payload, confirmed=True):
    settings.RUNTIME_DATA_DIR = str(tmp_path / "resumes")
    base = tmp_path / ("resumes" if kind == "cv" else "jobs") / RUN / ("resumes" if kind == "cv" else "jobs")
    base.mkdir(parents=True, exist_ok=True)
    if confirmed:
        (base / ("runtime_parsed_cv.json" if kind == "cv" else "runtime_parsed_jd.json")).write_text(json.dumps(payload), encoding="utf-8")
        (base / "confirm_override.json").write_text(json.dumps({"document_id": payload["id"]}), encoding="utf-8")
    return base

def test_cv_runtime_preparation_and_persistence(tmp_path):
    payload = {"id":"cv1", "domain":"General", "parsed_data":{"skills":["Kubernetes"],"experience_years":2,"projects":[],"work_experience":[{"company":"x","description":"Built systems."}]}}
    base = setup_runtime(tmp_path, "cv", payload)
    out = prepare_cv_runtime(RUN, "cv1", "IT", FakeEmb())
    assert out["normalized"]["domain"] == "IT"
    assert "Kubernetes" in out["normalized"]["skills"]["all"]
    assert any(x["skill"] == "Kubernetes" for x in out["embedding_artifact"]["skills"])
    assert (base / "prepared" / "embeddings_cv.pkl").is_file()
    manifest = json.loads((base / "prepared" / "preparation_manifest.json").read_text())
    assert manifest["document_id"] == "cv1" and manifest["run_id"] == RUN

def test_jd_runtime_manual_preferred_and_sparse(tmp_path):
    payload = {"id":"jd1", "parsed_data":{"job_title":"Role","preferred_skills":["PyTorch"],"required_skills":[],"responsibilities":[]}}
    setup_runtime(tmp_path, "jd", payload)
    out = prepare_jd_runtime(RUN, "jd1", "IT", FakeEmb())
    assert out["normalized"]["skills"]["preferred"] == ["PyTorch"]
    assert out["features"]["preferred_skills"] == ["PyTorch"]
    assert out["embedding_artifact"]["preferred_skills"][0]["skill"] == "PyTorch"
    assert out["chunks"] == [] and out["embedding_artifact"]["profile"]["vector"] is not None

def test_validation_and_confirmation_guards(tmp_path):
    payload = {"id":"cv1", "parsed_data":{}}
    setup_runtime(tmp_path, "cv", payload, confirmed=False)
    with pytest.raises(RuntimePreparationError): prepare_cv_runtime(RUN, "cv1", "IT", FakeEmb())
    with pytest.raises(RuntimePreparationError): prepare_cv_runtime(RUN, "cv1", "", FakeEmb())
    with pytest.raises(RuntimePreparationError): prepare_cv_runtime("../" + RUN, "cv1", "IT", FakeEmb())
    setup_runtime(tmp_path, "cv", payload)
    with pytest.raises(RuntimePreparationError): prepare_cv_runtime(RUN, "wrong", "IT", FakeEmb())

def test_reprepare_hash_deterministic(tmp_path):
    payload = {"id":"cv1", "parsed_data":{"skills":[]}}
    setup_runtime(tmp_path, "cv", payload)
    a = prepare_cv_runtime(RUN, "cv1", "IT", FakeEmb())
    b = prepare_cv_runtime(RUN, "cv1", "IT", FakeEmb())
    assert a["manifest"]["runtime_input_sha256"] == b["manifest"]["runtime_input_sha256"]
