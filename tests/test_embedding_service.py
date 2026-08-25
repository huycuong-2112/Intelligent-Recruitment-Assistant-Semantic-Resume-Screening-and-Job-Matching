import math
import pickle

import pytest

from src.Representation.embedding_service import EmbeddingService


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 6

    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            tokens = text.casefold().split()
            row = [float(sum(token in tokens for token in group)) for group in (("machine", "ai"), ("learning", "models"), ("python",), ("deep",), ("restaurant",), ("furniture",))]
            norm = math.sqrt(sum(value * value for value in row)) or 1.0
            rows.append([value / norm for value in row] if kwargs.get("normalize_embeddings") else row)
        return rows


def test_embedding_shapes_numerical_validity_and_similarity():
    service = EmbeddingService(); service._model = FakeModel()
    one = service.embed_text("Developed machine learning models using Python.")
    batch = service.embed_batch(["Developed machine learning models using Python.", "Built AI models with Python and deep learning.", "Designed a restaurant interior and selected furniture."])
    assert len(one) == service.dimension == 6 and len(batch) == 3
    assert all(math.isfinite(value) for row in batch for value in row)
    similarity = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert similarity(batch[0], batch[0]) > 0.99
    assert similarity(batch[0], batch[1]) > similarity(batch[0], batch[2])


@pytest.mark.integration
def test_real_embedding_backend_and_round_trip(tmp_path):
    service = EmbeddingService()
    texts = ["Developed machine learning models using Python.", "Built AI models with Python and deep learning.", "Designed a restaurant interior and selected furniture."]
    vectors = service.embed_batch(texts)
    assert len(vectors) == 3 and len(vectors[0]) == service.dimension
    assert all(math.isfinite(value) for row in vectors for value in row)
    cosine = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert cosine(vectors[0], vectors[0]) > 0.99
    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])
    artifact = {"id": "cv_test", "model": {"name": service.model_name, "dimension": service.dimension}, "text": texts[0], "source": "project", "vector": vectors[0]}
    path = tmp_path / "embedding.pkl"
    with path.open("wb") as handle: pickle.dump(artifact, handle)
    with path.open("rb") as handle: restored = pickle.load(handle)
    assert restored["id"] == artifact["id"] and restored["model"] == artifact["model"]
    assert restored["text"] == artifact["text"] and restored["source"] == artifact["source"]
    assert restored["vector"] == pytest.approx(artifact["vector"])
