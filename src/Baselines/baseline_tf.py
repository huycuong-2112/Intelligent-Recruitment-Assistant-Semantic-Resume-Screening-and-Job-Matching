from __future__ import annotations

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfBaseline:
    """
    TF-IDF lexical baseline.

    TF-IDF is fitted once on the full retrieval corpus:
    [JD, CV_1, CV_2, ..., CV_N]
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

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

        corpus = [jd_text, *cv_texts]

        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        jd_vector = tfidf_matrix[0]
        cv_vectors = tfidf_matrix[1:]

        scores = cosine_similarity(
            cv_vectors,
            jd_vector,
        ).reshape(-1)

        return scores