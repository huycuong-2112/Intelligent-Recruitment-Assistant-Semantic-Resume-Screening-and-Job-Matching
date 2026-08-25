"""Baseline semantic retrieval dùng all-MiniLM-L6-v2 và FAISS inner product.

Embedding được chuẩn hóa L2 trước khi thêm vào ``IndexFlatIP``. Vì vậy inner
product của FAISS chính là cosine similarity. Mô-đun có một index NumPy tối giản
chỉ làm phương án dự phòng để có thể import và kiểm thử logic dữ liệu khi gói
``faiss-cpu`` chưa được cài; khi FAISS khả dụng, index FAISS luôn được dùng.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


Candidate = Union[Mapping[str, Any], str]


class _NumpyIndexFlatIP:
    """Thay thế nội bộ tối thiểu cho FAISS IndexFlatIP khi FAISS không có sẵn."""

    def __init__(self, dimension: int) -> None:
        """Khởi tạo index inner-product rỗng với số chiều xác định.

        Args:
            dimension: Số chiều của embedding. Phải lớn hơn 0.

        Raises:
            ValueError: Nếu số chiều không dương.
        """
        if dimension <= 0:
            raise ValueError("Số chiều vector phải lớn hơn 0.")
        self.d = int(dimension)
        self._vectors = np.empty((0, self.d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        """Trả về số vector đang được lưu trong index."""
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        """Thêm một ma trận vector vào index.

        Args:
            vectors: Ma trận float có kích thước ``(n_vectors, d)``.

        Raises:
            ValueError: Nếu vector không phải ma trận hai chiều hoặc sai số chiều.
        """
        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.ndim != 2 or vector_array.shape[1] != self.d:
            raise ValueError("Vector thêm vào index có số chiều không khớp.")
        if vector_array.shape[0] > 0:
            self._vectors = np.vstack((self._vectors, vector_array))

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tìm ``k`` vector có inner product lớn nhất cho từng truy vấn.

        Args:
            queries: Ma trận truy vấn kích thước ``(n_queries, d)``.
            k: Số lượng kết quả cần lấy cho mỗi truy vấn.

        Returns:
            Cặp ``(scores, indices)`` theo quy ước của FAISS.

        Raises:
            ValueError: Nếu truy vấn có số chiều không hợp lệ.
        """
        query_array = np.asarray(queries, dtype=np.float32)
        if query_array.ndim != 2 or query_array.shape[1] != self.d:
            raise ValueError("Vector truy vấn có số chiều không khớp.")

        result_k = max(0, int(k))
        scores = np.full(
            (query_array.shape[0], result_k), -np.inf, dtype=np.float32
        )
        indices = np.full((query_array.shape[0], result_k), -1, dtype=np.int64)
        if result_k == 0 or self.ntotal == 0:
            return scores, indices

        similarities = query_array @ self._vectors.T
        available = min(result_k, self.ntotal)
        ranked_indices = np.argsort(-similarities, axis=1, kind="stable")[:, :available]
        ranked_scores = np.take_along_axis(similarities, ranked_indices, axis=1)
        scores[:, :available] = ranked_scores
        indices[:, :available] = ranked_indices
        return scores, indices


class MiniLMBaselineEngine:
    """Xây dựng và truy vấn baseline dense retrieval cho JD và CV.

    Mặc định engine tải model ``sentence-transformers/all-MiniLM-L6-v2`` và tạo
    index ``faiss.IndexFlatIP(384)``. Có thể truyền model hoặc index đã tạo sẵn
    để kiểm thử, chạy offline hoặc tái sử dụng tài nguyên trong ứng dụng.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_DIMENSION = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        index: Optional[Any] = None,
    ) -> None:
        """Khởi tạo model MiniLM và index inner-product rỗng.

        Args:
            model_name: Tên hoặc đường dẫn Hugging Face của SentenceTransformer.
            device: Thiết bị model, ví dụ ``"cpu"`` hoặc ``"cuda"``.
            model: Model tương thích có phương thức ``encode`` để tiêm từ ngoài.
            index: Index tương thích FAISS để tiêm từ ngoài.
        """
        self.model_name = model_name
        self.device = device
        self.model = model if model is not None else self._create_model()
        self.embedding_dimension = self._get_embedding_dimension()
        self.index = index if index is not None else self._new_index()
        self._indexed_candidates: List[Candidate] = []

    def _create_model(self) -> Optional[Any]:
        """Tải SentenceTransformer mặc định nếu dependency đang khả dụng.

        Returns:
            Instance SentenceTransformer, hoặc ``None`` khi thư viện chưa cài.

        Raises:
            RuntimeError: Nếu thư viện có mặt nhưng model không thể được tải.
        """
        if SentenceTransformer is None:
            return None
        try:
            if self.device is None:
                return SentenceTransformer(self.model_name)
            return SentenceTransformer(self.model_name, device=self.device)
        except Exception as error:
            raise RuntimeError(
                f"Không thể tải SentenceTransformer '{self.model_name}'."
            ) from error

    def _get_embedding_dimension(self) -> int:
        """Lấy số chiều embedding từ model và dùng 384 khi không xác định được.

        Returns:
            Số chiều embedding dương.
        """
        dimension_getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(dimension_getter):
            try:
                dimension = int(dimension_getter())
                if dimension > 0:
                    return dimension
            except (TypeError, ValueError):
                pass
        return self.DEFAULT_EMBEDDING_DIMENSION

    def _new_index(self) -> Any:
        """Tạo index FAISS ``IndexFlatIP`` hoặc fallback NumPy cùng số chiều.

        Returns:
            Index rỗng hỗ trợ ``add``, ``search``, ``ntotal`` và ``d``.
        """
        if faiss is not None:
            return faiss.IndexFlatIP(self.embedding_dimension)
        return _NumpyIndexFlatIP(self.embedding_dimension)

    @staticmethod
    def _normalise_vectors(vectors: Any) -> np.ndarray:
        """Chuẩn hóa L2 cho từng hàng của ma trận embedding.

        Hàng vector bằng 0 được giữ nguyên bằng 0, tránh chia cho 0 và đảm bảo
        kết quả luôn là ``float32`` phù hợp với FAISS.

        Args:
            vectors: Ma trận embedding hoặc một vector đơn.

        Returns:
            Ma trận embedding hai chiều đã chuẩn hóa L2.

        Raises:
            ValueError: Nếu đầu vào không thể biểu diễn thành vector hai chiều.
        """
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Embedding phải có thể chuyển thành mảng số.") from error

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("Embedding phải là vector hoặc ma trận hai chiều.")
        if matrix.shape[1] == 0:
            raise ValueError("Embedding không được có số chiều bằng 0.")

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalised = np.zeros_like(matrix, dtype=np.float32)
        non_zero_rows = norms[:, 0] > 0.0
        normalised[non_zero_rows] = (
            matrix[non_zero_rows] / norms[non_zero_rows]
        )
        return normalised.astype(np.float32, copy=False)

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Chuyển văn bản đầu vào thành chuỗi an toàn cho SentenceTransformer.

        Args:
            value: Văn bản thô hoặc ``None``.

        Returns:
            Chuỗi đã strip; ``None`` trở thành chuỗi rỗng.
        """
        return "" if value is None else str(value).strip()

    @classmethod
    def _candidate_text(cls, candidate: Candidate) -> str:
        """Lấy trường ``text`` từ bản ghi CV hoặc dùng trực tiếp chuỗi CV.

        Args:
            candidate: Dictionary CV có trường ``text`` hoặc chuỗi CV.

        Returns:
            Văn bản CV an toàn, có thể là chuỗi rỗng.
        """
        if isinstance(candidate, Mapping):
            return cls._coerce_text(candidate.get("text", ""))
        return cls._coerce_text(candidate)

    def encode_text(
        self,
        texts: Union[Optional[Any], Sequence[Optional[Any]]],
    ) -> np.ndarray:
        """Mã hóa một hoặc nhiều văn bản và chuẩn hóa L2 embedding đầu ra.

        Args:
            texts: Chuỗi đơn, sequence chuỗi, hoặc ``None``.

        Returns:
            Ma trận ``float32`` kích thước ``(n_texts, embedding_dimension)``.

        Raises:
            RuntimeError: Nếu sentence-transformers chưa được cài hoặc model chưa
                được cung cấp.
            ValueError: Nếu model trả embedding sai số chiều.
        """
        if isinstance(texts, (str, bytes)) or texts is None:
            text_list = [self._coerce_text(texts)]
        else:
            text_list = [self._coerce_text(text) for text in texts]

        if not text_list:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        if self.model is None:
            raise RuntimeError(
                "sentence-transformers chưa sẵn sàng. Hãy cài dependency hoặc "
                "truyền model tương thích vào MiniLMBaselineEngine."
            )

        encode_method = getattr(self.model, "encode", None)
        if not callable(encode_method):
            raise RuntimeError("Model MiniLM phải cung cấp phương thức encode.")
        try:
            embeddings = encode_method(
                text_list,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            embeddings = encode_method(text_list)

        normalised = self._normalise_vectors(embeddings)
        if normalised.shape[1] != self.embedding_dimension:
            raise ValueError(
                "Số chiều embedding từ model không khớp với số chiều index."
            )
        return normalised

    def build_index(self, cv_list: Optional[Sequence[Candidate]]) -> Any:
        """Mã hóa CV và xây dựng mới index inner-product cho toàn bộ danh sách.

        Args:
            cv_list: Danh sách CV dạng dictionary có khóa ``text`` hoặc chuỗi.

        Returns:
            FAISS ``IndexFlatIP`` (hoặc fallback tương thích khi FAISS thiếu).
        """
        self._indexed_candidates = list(cv_list) if cv_list is not None else []
        self.index = self._new_index()
        if not self._indexed_candidates:
            return self.index

        cv_texts = [
            self._candidate_text(candidate)
            for candidate in self._indexed_candidates
        ]
        vectors = self.encode_text(cv_texts)
        self.index.add(vectors)
        return self.index

    @staticmethod
    def _index_size(index: Any) -> int:
        """Lấy số vector trong index một cách an toàn.

        Args:
            index: Index FAISS hoặc index tương thích.

        Returns:
            Số vector không âm; trả 0 khi index không hợp lệ.
        """
        try:
            return max(0, int(getattr(index, "ntotal")))
        except (TypeError, ValueError, AttributeError):
            return 0

    def rank_candidates(
        self,
        jd_text: Optional[Any],
        cv_list: Optional[Sequence[Candidate]] = None,
        index: Optional[Any] = None,
        top_k: int = 10,
    ) -> List[Tuple[Candidate, float]]:
        """Tìm các CV tương đồng nhất với JD theo cosine similarity.

        Nếu không truyền ``cv_list`` hoặc ``index``, engine dùng dữ liệu gần nhất
        từ ``build_index``. Kết quả chỉ gồm index hợp lệ để không trả về sentinel
        ``-1`` của FAISS.

        Args:
            jd_text: Nội dung JD cần truy vấn.
            cv_list: Danh sách CV tương ứng với index; mặc định dùng CV đã index.
            index: Index cần truy vấn; mặc định dùng index nội bộ.
            top_k: Số CV tối đa cần trả về.

        Returns:
            Danh sách tuple ``(cv, cosine_similarity)`` giảm dần theo điểm.
        """
        if jd_text is None or not self._coerce_text(jd_text):
            return []
        try:
            requested_k = int(top_k)
        except (TypeError, ValueError):
            return []
        if requested_k <= 0:
            return []

        candidates = (
            list(cv_list) if cv_list is not None else self._indexed_candidates
        )
        active_index = index if index is not None else self.index
        index_size = self._index_size(active_index)
        if not candidates or index_size == 0:
            return []

        search_k = min(requested_k, len(candidates), index_size)
        jd_vector = self.encode_text(self._coerce_text(jd_text))
        scores, indices = active_index.search(jd_vector, search_k)
        ranked: List[Tuple[Candidate, float]] = []
        for score, candidate_index in zip(scores[0], indices[0]):
            numeric_index = int(candidate_index)
            if 0 <= numeric_index < len(candidates):
                ranked.append((candidates[numeric_index], float(score)))
        return ranked

    def get_scores_for_metrics(
        self,
        jd_text: Optional[Any],
        cv_list: Optional[Sequence[Candidate]] = None,
        index: Optional[Any] = None,
    ) -> np.ndarray:
        """Trả điểm cosine theo đúng thứ tự CV để dùng với các hàm metrics.

        Args:
            jd_text: Nội dung JD cần truy vấn.
            cv_list: Danh sách CV tương ứng index; mặc định dùng CV đã index.
            index: Index cần truy vấn; mặc định dùng index nội bộ.

        Returns:
            Vector điểm ``float32`` có cùng chiều dài với danh sách CV. Danh sách
            rỗng hoặc JD rỗng trả về vector rỗng hoặc vector 0 tương ứng.
        """
        candidates = (
            list(cv_list) if cv_list is not None else self._indexed_candidates
        )
        output = np.zeros(len(candidates), dtype=np.float32)
        if not candidates or jd_text is None or not self._coerce_text(jd_text):
            return output

        active_index = index if index is not None else self.index
        index_size = self._index_size(active_index)
        if index_size == 0:
            return output

        search_k = min(len(candidates), index_size)
        jd_vector = self.encode_text(self._coerce_text(jd_text))
        scores, indices = active_index.search(jd_vector, search_k)
        for score, candidate_index in zip(scores[0], indices[0]):
            numeric_index = int(candidate_index)
            if 0 <= numeric_index < len(candidates):
                output[numeric_index] = float(score)
        return output


__all__: List[str] = ["MiniLMBaselineEngine"]
