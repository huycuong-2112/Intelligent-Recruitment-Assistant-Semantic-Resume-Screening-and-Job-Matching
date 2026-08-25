"""Single boundary for loading and executing the configured local embedder."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EmbeddingService:
    def __init__(self, model_name: str | None = None, normalize_embeddings: bool = True, batch_size: int = 32) -> None:
        self.model_name = model_name or self._configured_model_name()
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self._model: Any = None

    @staticmethod
    def _configured_model_name() -> str:
        try:
            import yaml
            config_path = Path(__file__).resolve().parents[2] / "configs" / "mdms.yaml"
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            name = data.get("embedding", {}).get("model_name")
            if isinstance(name, str) and name.strip():
                return name
        except (ImportError, OSError, ValueError):
            pass
        raise RuntimeError("Embedding model_name is missing from configs/mdms.yaml")

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for local embeddings") from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def embed_text(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Cannot embed blank text")
        vector = self.model.encode([text], normalize_embeddings=self.normalize_embeddings, convert_to_numpy=True)[0]
        return [float(value) for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cannot embed blank text")
        if not texts:
            return []
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings, convert_to_numpy=True)
        return [[float(value) for value in row] for row in vectors]


def build_cv_profile_text(features: Any) -> str:
    education = features.education or {}
    parts = []
    if education.get("field"): parts.append(f"Education field: {education['field']}.")
    if features.profile.get("job_titles"): parts.append(f"Roles: {', '.join(features.profile['job_titles'])}.")
    if features.skills: parts.append(f"Skills: {', '.join(features.skills)}.")
    evidence = []
    for item in features.project_evidence + features.work_evidence:
        if isinstance(item, str): evidence.append(item)
        elif isinstance(item, dict): evidence.append(str(item.get("description") or item.get("summary") or item.get("details") or ""))
    for project in features.projects:
        if isinstance(project, dict):
            evidence.append(str(project.get("description") or project.get("summary") or project.get("details") or ""))
    if evidence: parts.append("Evidence: " + " ".join(evidence))
    return " ".join(parts).strip()


def build_jd_profile_text(features: Any) -> str:
    role = features.role or {}
    parts = []
    if role.get("job_title"): parts.append(f"Role: {role['job_title']}.")
    if role.get("overview"): parts.append(str(role["overview"]))
    if features.required_skills: parts.append(f"Required skills: {', '.join(features.required_skills)}.")
    if features.preferred_skills: parts.append(f"Preferred skills: {', '.join(features.preferred_skills)}.")
    if features.responsibilities: parts.append("Responsibilities: " + " ".join(str(x) for x in features.responsibilities))
    return " ".join(parts).strip()
