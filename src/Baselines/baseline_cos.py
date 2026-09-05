from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingCosineBaseline:
    """
    Dense semantic baseline:
    canonical document text
        -> pretrained MiniLM embedding
        -> cosine similarity

    No training.
    No Ground Truth.
    No MDMS.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def score(
        self,
        jd_text: str,
        cv_texts: list[str],
    ) -> np.ndarray:

        if not jd_text.strip():
            raise ValueError("JD text is empty")

        if not cv_texts:
            raise ValueError("No CV texts provided")

        if any(not text.strip() for text in cv_texts):
            raise ValueError("One or more CV texts are empty")

        texts = [jd_text, *cv_texts]

        # normalize_embeddings=True means dot product == cosine similarity
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        jd_embedding = embeddings[0]
        cv_embeddings = embeddings[1:]

        scores = cv_embeddings @ jd_embedding

        return scores