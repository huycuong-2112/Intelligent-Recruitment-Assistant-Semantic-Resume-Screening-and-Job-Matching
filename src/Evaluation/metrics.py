"""Các chỉ số đánh giá chất lượng xếp hạng CV và độ đồng thuận gán nhãn."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import ndcg_score


DEFAULT_RELEVANCE_THRESHOLD = 2.0
"""Ngưỡng relevance mặc định: nhãn từ 2.0 trở lên được xem là phù hợp."""

threshold = DEFAULT_RELEVANCE_THRESHOLD
"""Bí danh tương thích ngược cho ngưỡng relevance mặc định."""


def _as_1d_float_array(values: Sequence[Any], name: str) -> np.ndarray:
    """Chuyển dãy điểm thành vector float một chiều có kiểm soát NaN.

    Args:
        values: Dãy giá trị số.
        name: Tên dãy để đưa vào thông báo lỗi.

    Returns:
        Vector ``float64`` một chiều.

    Raises:
        ValueError: Nếu dãy không một chiều hoặc chứa giá trị không đổi được sang số.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} phải là dãy giá trị số.") from error
    if array.ndim != 1:
        raise ValueError(f"{name} phải là dãy một chiều.")
    return array


def _validate_ranking_inputs(
    y_true_rel: Sequence[Any],
    y_score: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Xác thực hai vector relevance và điểm dự đoán cho bài toán xếp hạng.

    Args:
        y_true_rel: Nhãn relevance thực tế.
        y_score: Điểm dự đoán của hệ thống.

    Returns:
        Cặp vector float đã xử lý NaN ở điểm dự đoán thành ``-inf``.

    Raises:
        ValueError: Nếu hai vector khác độ dài hoặc không hợp lệ.
    """
    relevance = _as_1d_float_array(y_true_rel, "y_true_rel")
    scores = _as_1d_float_array(y_score, "y_score")
    if relevance.size != scores.size:
        raise ValueError("y_true_rel và y_score phải có cùng số phần tử.")
    safe_scores = np.nan_to_num(
        scores,
        nan=-np.finfo(np.float64).max,
        posinf=np.finfo(np.float64).max,
        neginf=-np.finfo(np.float64).max,
    )
    return relevance, safe_scores


def _valid_k(k: Any, number_of_items: int) -> int:
    """Chuẩn hóa K về số lượng phần tử có thể đánh giá.

    Args:
        k: Giá trị K người gọi yêu cầu.
        number_of_items: Tổng số ứng viên có thể xếp hạng.

    Returns:
        K nguyên thuộc đoạn từ 0 đến ``number_of_items``.
    """
    try:
        requested_k = int(k)
    except (TypeError, ValueError):
        return 0
    return min(max(0, requested_k), max(0, number_of_items))


def _ranked_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Lấy chỉ số top-K giảm dần, ổn định khi các điểm bằng nhau.

    Args:
        scores: Vector điểm dự đoán.
        k: Số chỉ số cần lấy.

    Returns:
        Vector chỉ số xếp hạng giảm dần.
    """
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(-scores, kind="stable")[:k]


def precision_at_k(
    y_true_rel: Sequence[Any],
    y_score: Sequence[Any],
    k: int,
    rel_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """Tính Precision@K theo ngưỡng relevance.

    Args:
        y_true_rel: Nhãn relevance thực tế của các CV.
        y_score: Điểm dự đoán theo cùng thứ tự CV.
        k: Số CV đầu bảng cần đánh giá.
        rel_threshold: Nhãn tối thiểu để xem là CV phù hợp.

    Returns:
        Precision@K, hoặc 0.0 khi danh sách rỗng hay ``k`` không dương.
    """
    relevance, scores = _validate_ranking_inputs(y_true_rel, y_score)
    actual_k = _valid_k(k, relevance.size)
    if actual_k == 0:
        return 0.0
    top_indices = _ranked_indices(scores, actual_k)
    relevant_count = np.sum(relevance[top_indices] >= rel_threshold)
    return float(relevant_count / actual_k)


def recall_at_k(
    y_true_rel: Sequence[Any],
    y_score: Sequence[Any],
    k: int,
    rel_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """Tính Recall@K theo ngưỡng relevance.

    Args:
        y_true_rel: Nhãn relevance thực tế của các CV.
        y_score: Điểm dự đoán theo cùng thứ tự CV.
        k: Số CV đầu bảng cần đánh giá.
        rel_threshold: Nhãn tối thiểu để xem là CV phù hợp.

    Returns:
        Recall@K, hoặc 0.0 khi không có CV phù hợp hay ``k`` không dương.
    """
    relevance, scores = _validate_ranking_inputs(y_true_rel, y_score)
    relevant_total = int(np.sum(relevance >= rel_threshold))
    actual_k = _valid_k(k, relevance.size)
    if relevant_total == 0 or actual_k == 0:
        return 0.0
    top_indices = _ranked_indices(scores, actual_k)
    relevant_count = int(np.sum(relevance[top_indices] >= rel_threshold))
    return float(relevant_count / relevant_total)


def compute_ndcg_at_k(
    y_true_rel: Sequence[Any],
    y_score: Sequence[Any],
    k: int,
) -> float:
    """Tính Normalized Discounted Cumulative Gain tại K.

    Hàm sử dụng gain relevance tuyến tính, tương thích với ``sklearn.ndcg_score``
    trong baseline trước đó. Relevance âm được chặn về 0 để NDCG luôn có ý nghĩa
    và nằm trong miền ``[0, 1]``.

    Args:
        y_true_rel: Nhãn relevance thực tế.
        y_score: Điểm dự đoán theo cùng thứ tự.
        k: Số kết quả đầu cần đánh giá.

    Returns:
        NDCG@K trong khoảng từ 0.0 đến 1.0.
    """
    relevance, scores = _validate_ranking_inputs(y_true_rel, y_score)
    actual_k = _valid_k(k, relevance.size)
    if actual_k == 0:
        return 0.0

    gains = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
    gains = np.maximum(gains, 0.0)
    if not np.any(gains > 0.0):
        return 0.0

    score = float(ndcg_score([gains], [scores], k=actual_k))
    if not math.isfinite(score):
        return 0.0
    return float(min(1.0, max(0.0, score)))


def ndcg_at_k(
    y_true_rel: Sequence[Any],
    y_score: Sequence[Any],
    k: int,
) -> float:
    """Bí danh tương thích ngược của :func:`compute_ndcg_at_k`.

    Args:
        y_true_rel: Nhãn relevance thực tế.
        y_score: Điểm dự đoán theo cùng thứ tự.
        k: Số kết quả đầu cần đánh giá.

    Returns:
        NDCG@K trong khoảng từ 0.0 đến 1.0.
    """
    return compute_ndcg_at_k(y_true_rel, y_score, k)


def mrr_score(
    y_true_rel: Sequence[Any],
    y_score: Sequence[Any],
    rel_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> float:
    """Tính Mean Reciprocal Rank cho một truy vấn JD.

    MRR của một truy vấn là nghịch đảo thứ hạng CV phù hợp đầu tiên. Để lấy MRR
    của cả tập JD, gọi hàm cho từng JD rồi lấy trung bình các kết quả.

    Args:
        y_true_rel: Nhãn relevance thực tế của các CV.
        y_score: Điểm dự đoán theo cùng thứ tự CV.
        rel_threshold: Nhãn tối thiểu để xem là CV phù hợp.

    Returns:
        Reciprocal rank đầu tiên, hoặc 0.0 nếu không có CV phù hợp.
    """
    relevance, scores = _validate_ranking_inputs(y_true_rel, y_score)
    if relevance.size == 0:
        return 0.0
    for rank, index in enumerate(_ranked_indices(scores, relevance.size), start=1):
        if relevance[index] >= rel_threshold:
            return float(1.0 / rank)
    return 0.0


def _label_key(value: Any) -> Hashable:
    """Chuyển nhãn bất kỳ thành khóa hashable dùng trong bảng đếm Cohen's Kappa.

    Args:
        value: Nhãn do một người đánh giá cung cấp.

    Returns:
        Giá trị hashable giữ nguyên khi có thể, hoặc biểu diễn ``repr`` an toàn.
    """
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _cohen_kappa(rater_a: Sequence[Any], rater_b: Sequence[Any]) -> float:
    """Tính Cohen's Kappa không trọng số cho hai người đánh giá.

    Args:
        rater_a: Chuỗi nhãn của người đánh giá thứ nhất.
        rater_b: Chuỗi nhãn của người đánh giá thứ hai.

    Returns:
        Cohen's Kappa trong ``[-1, 1]``; trả 0.0 cho danh sách rỗng.

    Raises:
        ValueError: Nếu hai người đánh giá có số lượng nhãn khác nhau.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Các người đánh giá phải có cùng số lượng nhãn.")
    sample_count = len(rater_a)
    if sample_count == 0:
        return 0.0

    labels_a = [_label_key(value) for value in rater_a]
    labels_b = [_label_key(value) for value in rater_b]
    observed_agreement = sum(
        label_a == label_b for label_a, label_b in zip(labels_a, labels_b)
    ) / sample_count
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    all_labels = set(counts_a).union(counts_b)
    expected_agreement = sum(
        (counts_a[label] / sample_count) * (counts_b[label] / sample_count)
        for label in all_labels
    )
    denominator = 1.0 - expected_agreement
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return 1.0 if math.isclose(observed_agreement, 1.0, abs_tol=1e-12) else 0.0
    return float(
        max(
            -1.0,
            min(1.0, (observed_agreement - expected_agreement) / denominator),
        )
    )


def _coerce_rater_matrix(
    ratings: Sequence[Sequence[Any]],
) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
    """Diễn giải ma trận rating dạng ba hàng hoặc ba cột thành ba rater.

    Args:
        ratings: Ma trận có đúng ba rater theo hàng hoặc theo cột.

    Returns:
        Ba dãy nhãn tương ứng ba người đánh giá.

    Raises:
        ValueError: Nếu ma trận không có đúng ba người đánh giá.
    """
    rows = [list(row) for row in ratings]
    if not rows:
        return [], [], []
    if len(rows) == 3:
        return rows[0], rows[1], rows[2]
    if rows and all(len(row) == 3 for row in rows):
        columns = list(zip(*rows))
        return list(columns[0]), list(columns[1]), list(columns[2])
    raise ValueError("Cần cung cấp đúng ba người đánh giá Ground Truth.")


def evaluate_inter_rater_agreement(
    rater_1: Sequence[Any],
    rater_2: Optional[Sequence[Any]] = None,
    rater_3: Optional[Sequence[Any]] = None,
) -> float:
    """Tính Cohen's Kappa trung bình trên ba cặp người đánh giá Ground Truth.

    Có thể gọi bằng ba vector ``(rater_1, rater_2, rater_3)`` hoặc một ma trận
    gồm đúng ba hàng/ba cột. Điểm cuối là trung bình của Kappa các cặp (1, 2),
    (1, 3) và (2, 3).

    Args:
        rater_1: Nhãn của rater 1 hoặc ma trận rating ba rater.
        rater_2: Nhãn của rater 2.
        rater_3: Nhãn của rater 3.

    Returns:
        Cohen's Kappa trung bình trong khoảng từ -1.0 đến 1.0.

    Raises:
        ValueError: Nếu không đủ ba rater hoặc số nhãn giữa các rater khác nhau.
    """
    if rater_2 is None and rater_3 is None:
        first, second, third = _coerce_rater_matrix(rater_1)  # type: ignore[arg-type]
    elif rater_2 is not None and rater_3 is not None:
        first, second, third = rater_1, rater_2, rater_3
    else:
        raise ValueError("Cần cung cấp đủ nhãn của cả ba người đánh giá.")

    kappas: List[float] = [
        _cohen_kappa(first, second),
        _cohen_kappa(first, third),
        _cohen_kappa(second, third),
    ]
    return float(sum(kappas) / len(kappas))


__all__: List[str] = [
    "DEFAULT_RELEVANCE_THRESHOLD",
    "compute_ndcg_at_k",
    "evaluate_inter_rater_agreement",
    "mrr_score",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
