"""Các hàm chấm điểm từng thành phần của MDMS.

Mỗi hàm trong mô-đun này trả về điểm đã được chặn trong khoảng ``[0.0, 1.0]``.
Các đầu vào dạng chuỗi, danh sách hoặc tri thức ESCO thiếu dữ liệu được xử lý an
toàn để một CV chưa trích xuất đủ thông tin không làm gián đoạn quá trình xếp hạng.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from numbers import Real
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np


DEGREE_RANKING: Dict[str, int] = {
    "none": 0,
    "high school": 1,
    "diploma": 2,
    "associate": 2,
    "bachelor": 3,
    "engineer": 3,
    "master": 4,
    "phd": 5,
    "None": 0,
    "High School": 1,
    "Diploma": 2,
    "Associate": 2,
    "Bachelor": 3,
    "Engineer": 3,
    "Master": 4,
    "PhD": 5,
    "Ph.D": 5,
}
"""Bảng quy đổi cấp bậc học vấn từ thấp đến cao dùng trong MDMS."""


def _clamp_score(value: float) -> float:
    """Giới hạn một giá trị số hữu hạn vào miền điểm ``[0.0, 1.0]``.

    Args:
        value: Giá trị cần chuẩn hóa.

    Returns:
        Giá trị số thực nằm trong khoảng từ 0.0 đến 1.0.
    """
    if not math.isfinite(value):
        return 0.0
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return 0.0
    if math.isclose(value, 1.0, abs_tol=1e-12):
        return 1.0
    return float(min(1.0, max(0.0, value)))


def _normalise_text(value: Any) -> str:
    """Chuẩn hóa văn bản để so khớp không phân biệt hoa thường và dấu câu.

    Hàm bảo toàn ngữ nghĩa thường gặp của các kỹ năng như ``C++``, ``C#`` và
    ``.NET`` trước khi loại bỏ ký tự đặc biệt.

    Args:
        value: Giá trị văn bản hoặc giá trị bất kỳ có thể chuyển thành chuỗi.

    Returns:
        Chuỗi đã chuẩn hóa; trả về chuỗi rỗng khi đầu vào là ``None``.
    """
    if value is None:
        return ""

    text = str(value).strip().casefold()
    if not text:
        return ""

    replacements: Dict[str, str] = {
        "c++": " cpp ",
        "c#": " csharp ",
        ".net": " dotnet ",
        "node.js": " nodejs ",
        "node js": " nodejs ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _as_text_list(values: Optional[Iterable[Any]]) -> List[str]:
    """Chuyển đầu vào đơn hoặc tập đầu vào thành danh sách chuỗi chuẩn hóa duy nhất.

    Args:
        values: Chuỗi đơn, iterable các giá trị, hoặc ``None``.

    Returns:
        Danh sách chuỗi không rỗng, giữ nguyên thứ tự xuất hiện đầu tiên.
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raw_values: Iterable[Any] = [values]
    else:
        try:
            raw_values = values
        except TypeError:
            raw_values = [values]

    normalised_values: List[str] = []
    seen: Set[str] = set()
    try:
        iterator = iter(raw_values)
    except TypeError:
        iterator = iter([raw_values])

    for value in iterator:
        normalised = _normalise_text(value)
        if normalised and normalised not in seen:
            seen.add(normalised)
            normalised_values.append(normalised)
    return normalised_values


def _iter_aliases(value: Any) -> List[str]:
    """Trích xuất các nhãn đồng nghĩa từ một giá trị ESCO linh hoạt.

    Args:
        value: Chuỗi, iterable chuỗi, hoặc ``None`` từ trường nhãn thay thế.

    Returns:
        Danh sách nhãn thô để tiếp tục chuẩn hóa.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[\n;|]", value) if item.strip()]
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        return [str(value).strip()]


def _register_esco_record(
    alias_map: Dict[str, str],
    canonical_value: Any,
    aliases: Iterable[Any],
) -> None:
    """Đăng ký một kỹ năng chuẩn và mọi nhãn thay thế vào bảng tra cứu.

    Args:
        alias_map: Bảng ánh xạ nhãn chuẩn hóa sang kỹ năng chuẩn hóa.
        canonical_value: Nhãn chuẩn của kỹ năng.
        aliases: Các nhãn đồng nghĩa của kỹ năng.
    """
    canonical = _normalise_text(canonical_value)
    if not canonical:
        return

    alias_map.setdefault(canonical, canonical)
    for alias in aliases:
        normalised_alias = _normalise_text(alias)
        if normalised_alias:
            alias_map.setdefault(normalised_alias, canonical)


def _build_esco_alias_map(esco_kb: Optional[Any]) -> Dict[str, str]:
    """Tạo bảng ánh xạ đồng nghĩa từ các cấu trúc ESCO thường dùng.

    Hỗ trợ cả danh sách bản ghi xuất từ ``esco_subset.json`` và dictionary dạng
    ``{"python": ["python programming"]}``. Với bản ghi dictionary, các trường
    ``skill_name``, ``preferredLabel``, ``name`` và ``alt_labels`` được nhận diện.

    Args:
        esco_kb: Knowledge base ESCO hoặc bảng đồng nghĩa do ứng dụng cung cấp.

    Returns:
        Dictionary ánh xạ nhãn đã chuẩn hóa sang nhãn kỹ năng chuẩn hóa.
    """
    alias_map: Dict[str, str] = {}
    if esco_kb is None:
        return alias_map

    if isinstance(esco_kb, Mapping):
        for key, value in esco_kb.items():
            if isinstance(value, Mapping):
                canonical = (
                    value.get("skill_name")
                    or value.get("preferredLabel")
                    or value.get("name")
                    or key
                )
                aliases: List[Any] = []
                for alias_key in ("alt_labels", "altLabels", "aliases", "synonyms"):
                    aliases.extend(_iter_aliases(value.get(alias_key)))
                aliases.append(key)
                _register_esco_record(alias_map, canonical, aliases)
            else:
                _register_esco_record(alias_map, key, _iter_aliases(value))
        return alias_map

    try:
        records = iter(esco_kb)
    except TypeError:
        return alias_map

    for record in records:
        if not isinstance(record, Mapping):
            continue
        canonical = (
            record.get("skill_name")
            or record.get("preferredLabel")
            or record.get("name")
            or record.get("label")
        )
        aliases = []
        for alias_key in ("alt_labels", "altLabels", "aliases", "synonyms"):
            aliases.extend(_iter_aliases(record.get(alias_key)))
        _register_esco_record(alias_map, canonical, aliases)
    return alias_map


def _canonical_skill_set(
    skills: Optional[Iterable[Any]],
    alias_map: Mapping[str, str],
) -> Set[str]:
    """Chuyển danh sách kỹ năng thành tập kỹ năng chuẩn theo ESCO.

    Args:
        skills: Danh sách kỹ năng cần chuẩn hóa.
        alias_map: Bảng đồng nghĩa ESCO đã xây dựng.

    Returns:
        Tập kỹ năng chuẩn hóa, không chứa chuỗi rỗng.
    """
    return {
        alias_map.get(skill, skill)
        for skill in _as_text_list(skills)
        if alias_map.get(skill, skill)
    }


def _coverage_score(candidate: Set[str], requested: Set[str]) -> float:
    """Tính tỷ lệ kỹ năng yêu cầu xuất hiện trong tập kỹ năng ứng viên.

    Args:
        candidate: Tập kỹ năng của ứng viên.
        requested: Tập kỹ năng của công việc.

    Returns:
        Tỷ lệ bao phủ; trả về 0.0 khi không có kỹ năng cần đánh giá.
    """
    if not requested:
        return 0.0
    return _clamp_score(len(candidate.intersection(requested)) / len(requested))


def skill_score(
    candidate_skills: Optional[Iterable[Any]],
    required_skills: Optional[Iterable[Any]],
    optional_skills: Optional[Iterable[Any]] = None,
    esco_kb: Optional[Any] = None,
    required_weight: float = 0.8,
    optional_weight: float = 0.2,
) -> float:
    """Tính điểm kỹ năng bắt buộc và tùy chọn có xét đồng nghĩa ESCO.

    Công thức khi JD có cả hai nhóm kỹ năng là:

    ``score = w_required * coverage_required + w_optional * coverage_optional``.

    Nếu một nhóm JD rỗng, trọng số của nhóm còn lại được chuẩn hóa thành 1.0 để
    không làm giảm điểm chỉ vì JD không khai báo kỹ năng tùy chọn.

    Args:
        candidate_skills: Các kỹ năng được trích xuất từ CV.
        required_skills: Các kỹ năng bắt buộc của JD.
        optional_skills: Các kỹ năng ưu tiên hoặc tùy chọn của JD.
        esco_kb: Tri thức ESCO mở rộng để nối các nhãn đồng nghĩa.
        required_weight: Trọng số dành cho kỹ năng bắt buộc.
        optional_weight: Trọng số dành cho kỹ năng tùy chọn.

    Returns:
        Điểm kỹ năng trong khoảng từ 0.0 đến 1.0.

    Raises:
        ValueError: Nếu trọng số âm hoặc tổng trọng số khả dụng bằng 0.
    """
    if required_weight < 0.0 or optional_weight < 0.0:
        raise ValueError("Trọng số kỹ năng không được âm.")

    alias_map = _build_esco_alias_map(esco_kb)
    candidate = _canonical_skill_set(candidate_skills, alias_map)
    required = _canonical_skill_set(required_skills, alias_map)
    optional = _canonical_skill_set(optional_skills, alias_map)

    components: List[float] = []
    weights: List[float] = []
    if required:
        components.append(_coverage_score(candidate, required))
        weights.append(required_weight)
    if optional:
        components.append(_coverage_score(candidate, optional))
        weights.append(optional_weight)

    if not components:
        return 0.0
    weight_total = sum(weights)
    if weight_total <= 0.0:
        raise ValueError("Tổng trọng số của các nhóm kỹ năng phải lớn hơn 0.")
    return _clamp_score(
        sum(component * weight for component, weight in zip(components, weights))
        / weight_total
    )


def _coerce_years(value: Optional[Any]) -> Optional[float]:
    """Chuyển dữ liệu số năm kinh nghiệm thành số thực không âm.

    Args:
        value: Số, chuỗi như ``"3+"`` hoặc ``"3 years"``, hay ``None``.

    Returns:
        Số năm kinh nghiệm, hoặc ``None`` nếu đầu vào không thể diễn giải.
    """
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        return max(0.0, numeric_value) if math.isfinite(numeric_value) else None

    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    return float(match.group())


def exp_score(
    candidate_years: Optional[Any],
    required_years: Optional[Any],
    overqualification_threshold: float = 5.0,
    overqualification_penalty: float = 0.8,
) -> float:
    """Tính điểm kinh nghiệm và phạt ứng viên vượt yêu cầu quá xa.

    Khi yêu cầu kinh nghiệm hợp lệ và dương, điểm nền là
    ``min(candidate_years / required_years, 1.0)``. Nếu ứng viên vượt yêu cầu
    hơn ``overqualification_threshold`` năm, điểm nền được nhân với hệ số phạt
    ``overqualification_penalty``. JD không nêu yêu cầu kinh nghiệm được xem là
    một tiêu chí trung tính và nhận điểm 1.0.

    Args:
        candidate_years: Số năm kinh nghiệm của ứng viên.
        required_years: Số năm kinh nghiệm JD yêu cầu.
        overqualification_threshold: Khoảng vượt ngưỡng trước khi áp dụng phạt.
        overqualification_penalty: Hệ số phạt, mặc định là 0.8.

    Returns:
        Điểm kinh nghiệm trong khoảng từ 0.0 đến 1.0.

    Raises:
        ValueError: Nếu ngưỡng âm hoặc hệ số phạt nằm ngoài ``[0, 1]``.
    """
    if overqualification_threshold < 0.0:
        raise ValueError("Ngưỡng overqualified không được âm.")
    if not 0.0 <= overqualification_penalty <= 1.0:
        raise ValueError("Hệ số phạt overqualified phải nằm trong khoảng [0, 1].")

    required = _coerce_years(required_years)
    candidate = _coerce_years(candidate_years)
    if required is None or required <= 0.0:
        return 1.0

    candidate_value = candidate if candidate is not None else 0.0
    score = min(candidate_value / required, 1.0)
    if candidate_value - required > overqualification_threshold:
        score *= overqualification_penalty
    return _clamp_score(score)


def _normalise_degree(value: Optional[Any]) -> Optional[str]:
    """Chuẩn hóa tên bằng cấp từ các cách viết tiếng Anh và tiếng Việt phổ biến.

    Args:
        value: Bằng cấp thô được trích xuất từ CV hoặc JD.

    Returns:
        Khóa của ``DEGREE_RANKING`` hoặc ``None`` khi không nhận diện được.
    """
    text = _normalise_text(value)
    if not text:
        return None

    patterns: List[tuple[str, tuple[str, ...]]] = [
        ("phd", ("phd", "ph d", "doctorate", "doctoral", "tien si")),
        ("master", ("master", "msc", "mba", "thac si")),
        ("bachelor", ("bachelor", "undergraduate", "cu nhan", "bsc", "ba")),
        ("engineer", ("engineer", "ky su")),
        ("associate", ("associate", "cao dang")),
        ("diploma", ("diploma", "certificate", "chung chi")),
        ("high school", ("high school", "secondary school", "trung hoc")),
        ("none", ("none", "no degree", "khong co")),
    ]
    for canonical, aliases in patterns:
        if any(alias in text for alias in aliases):
            return canonical
    return text if text in DEGREE_RANKING else None


def _cosine_similarity(left: Any, right: Any) -> Optional[float]:
    """Tính cosine similarity an toàn giữa hai vector embedding.

    Args:
        left: Vector thứ nhất.
        right: Vector thứ hai.

    Returns:
        Cosine similarity đã chặn về ``[0, 1]`` hoặc ``None`` nếu vector không
        tương thích hoặc có chuẩn L2 bằng 0.
    """
    try:
        left_vector = np.asarray(left, dtype=np.float64).reshape(-1)
        right_vector = np.asarray(right, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if left_vector.size == 0 or left_vector.shape != right_vector.shape:
        return None

    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    similarity = float(np.dot(left_vector, right_vector) / denominator)
    return _clamp_score(similarity)


def _fallback_string_similarity(left: str, right: str) -> float:
    """Ước lượng độ giống văn bản khi không có embedding engine.

    Args:
        left: Văn bản thứ nhất đã hoặc chưa chuẩn hóa.
        right: Văn bản thứ hai đã hoặc chưa chuẩn hóa.

    Returns:
        Độ tương đồng trong ``[0, 1]`` dựa trên Jaccard token và SequenceMatcher.
    """
    left_normalised = _normalise_text(left)
    right_normalised = _normalise_text(right)
    if not left_normalised or not right_normalised:
        return 0.0
    if left_normalised == right_normalised:
        return 1.0

    left_tokens = set(left_normalised.split())
    right_tokens = set(right_normalised.split())
    union = left_tokens.union(right_tokens)
    jaccard = len(left_tokens.intersection(right_tokens)) / len(union) if union else 0.0
    sequence_ratio = SequenceMatcher(None, left_normalised, right_normalised).ratio()
    return _clamp_score(max(jaccard, sequence_ratio))


def _embedding_text_similarity(
    left: str,
    right: str,
    embedding_engine: Optional[Any],
) -> float:
    """Tính similarity văn bản bằng engine embedding hoặc fallback chuỗi.

    Engine callable có thể trả trực tiếp một điểm hoặc cặp embedding. Engine có
    phương thức ``encode`` (ví dụ SentenceTransformer) được gọi với hai văn bản.

    Args:
        left: Văn bản thứ nhất.
        right: Văn bản thứ hai.
        embedding_engine: Engine embedding hoặc hàm similarity tùy chọn.

    Returns:
        Điểm tương đồng nằm trong ``[0, 1]``.
    """
    if not left or not right:
        return 0.0
    if _normalise_text(left) == _normalise_text(right):
        return 1.0

    if embedding_engine is not None:
        if callable(embedding_engine) and not hasattr(embedding_engine, "encode"):
            try:
                result = embedding_engine(left, right)
                if isinstance(result, Real):
                    return _clamp_score(float(result))
                result_array = np.asarray(result, dtype=np.float64)
                if result_array.ndim >= 2 and result_array.shape[0] >= 2:
                    similarity = _cosine_similarity(result_array[0], result_array[1])
                    if similarity is not None:
                        return similarity
            except (TypeError, ValueError, AttributeError):
                pass

        encode_method = getattr(embedding_engine, "encode", None)
        if callable(encode_method):
            try:
                embeddings = encode_method(
                    [left, right],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except TypeError:
                try:
                    embeddings = encode_method([left, right])
                except (TypeError, ValueError, AttributeError):
                    embeddings = None
            except (ValueError, AttributeError):
                embeddings = None

            if embeddings is not None:
                embeddings_array = np.asarray(embeddings)
                if embeddings_array.ndim >= 2 and embeddings_array.shape[0] >= 2:
                    similarity = _cosine_similarity(
                        embeddings_array[0], embeddings_array[1]
                    )
                    if similarity is not None:
                        return similarity

    return _fallback_string_similarity(left, right)


def edu_score(
    candidate_degree: Optional[Any],
    required_degree: Optional[Any],
    candidate_field: Optional[Any] = None,
    required_field: Optional[Any] = None,
    embedding_engine: Optional[Any] = None,
    degree_weight: float = 0.7,
    field_weight: float = 0.3,
) -> float:
    """Tính điểm học vấn từ cấp bậc bằng cấp và độ giống chuyên ngành.

    Điểm bằng cấp là 1.0 khi ứng viên có bậc học vấn bằng hoặc cao hơn yêu cầu;
    nếu thấp hơn, điểm là tỷ lệ ``rank_candidate / rank_required``. Điểm chuyên
    ngành dùng cosine embedding khi có engine; nếu không, dùng so khớp chuỗi.
    Hai thành phần được kết hợp theo trọng số và tự chuẩn hóa khi JD thiếu một
    trong hai yêu cầu.

    Args:
        candidate_degree: Bằng cấp cao nhất của ứng viên.
        required_degree: Bằng cấp JD yêu cầu.
        candidate_field: Chuyên ngành của ứng viên.
        required_field: Chuyên ngành JD yêu cầu.
        embedding_engine: SentenceTransformer hoặc engine tương thích.
        degree_weight: Trọng số thành phần cấp bậc bằng cấp.
        field_weight: Trọng số thành phần chuyên ngành.

    Returns:
        Điểm học vấn trong khoảng từ 0.0 đến 1.0.

    Raises:
        ValueError: Nếu một trọng số âm hoặc tổng trọng số khả dụng bằng 0.
    """
    if degree_weight < 0.0 or field_weight < 0.0:
        raise ValueError("Trọng số học vấn không được âm.")

    required_degree_key = _normalise_degree(required_degree)
    candidate_degree_key = _normalise_degree(candidate_degree)
    required_field_text = _normalise_text(required_field)
    candidate_field_text = _normalise_text(candidate_field)

    components: List[float] = []
    weights: List[float] = []
    if required_degree_key is not None:
        required_rank = DEGREE_RANKING[required_degree_key]
        candidate_rank = (
            DEGREE_RANKING[candidate_degree_key]
            if candidate_degree_key is not None
            else 0
        )
        degree_component = 1.0
        if required_rank > 0:
            degree_component = min(candidate_rank / required_rank, 1.0)
        components.append(_clamp_score(degree_component))
        weights.append(degree_weight)

    if required_field_text:
        field_component = _embedding_text_similarity(
            candidate_field_text,
            required_field_text,
            embedding_engine,
        )
        components.append(field_component)
        weights.append(field_weight)

    if not components:
        return 1.0
    weight_total = sum(weights)
    if weight_total <= 0.0:
        raise ValueError("Tổng trọng số học vấn phải lớn hơn 0.")
    return _clamp_score(
        sum(component * weight for component, weight in zip(components, weights))
        / weight_total
    )


def _token_cosine_similarity(left: str, right: str) -> float:
    """Tính cosine similarity giữa hai văn bản bằng vector tần suất token.

    Args:
        left: Văn bản thứ nhất.
        right: Văn bản thứ hai.

    Returns:
        Cosine similarity trong khoảng từ 0.0 đến 1.0.
    """
    left_tokens = _normalise_text(left).split()
    right_tokens = _normalise_text(right).split()
    if not left_tokens or not right_tokens:
        return 0.0

    left_counts = Counter(left_tokens)
    right_counts = Counter(right_tokens)
    dot_product = sum(
        left_counts[token] * right_counts.get(token, 0) for token in left_counts
    )
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return _clamp_score(dot_product / (left_norm * right_norm))


def _title_similarity(
    candidate_titles: Sequence[str],
    target_titles: Sequence[str],
    embedding_engine: Optional[Any],
) -> float:
    """Tính cosine similarity giữa nhóm chức danh ứng viên và JD.

    Args:
        candidate_titles: Các chức danh đã làm của ứng viên.
        target_titles: Chức danh mục tiêu hoặc chức danh liên quan trong JD.
        embedding_engine: Engine embedding tùy chọn.

    Returns:
        Điểm cosine trong ``[0, 1]``; trả về 0.0 khi một phía không có chức danh.
    """
    if not candidate_titles or not target_titles:
        return 0.0
    candidate_text = " ".join(candidate_titles)
    target_text = " ".join(target_titles)

    if embedding_engine is not None:
        return _embedding_text_similarity(
            candidate_text,
            target_text,
            embedding_engine,
        )
    return _token_cosine_similarity(candidate_text, target_text)


def domain_score(
    candidate_skills: Optional[Iterable[Any]],
    domain_skills: Optional[Iterable[Any]],
    candidate_job_titles: Optional[Iterable[Any]],
    target_job_titles: Optional[Iterable[Any]],
    embedding_engine: Optional[Any] = None,
    skill_weight: float = 0.5,
    title_weight: float = 0.5,
    esco_kb: Optional[Any] = None,
) -> float:
    """Tính điểm lĩnh vực từ Jaccard kỹ năng và cosine chức danh.

    Thành phần kỹ năng là ``|CV_skills ∩ JD_domain_skills| / |CV_skills ∪
    JD_domain_skills|``. Thành phần chức danh là cosine similarity của embedding
    hai tập chức danh; khi không có engine, cosine trên vector tần suất token được
    dùng thay thế. Trọng số của các thành phần JD không khai báo được chuẩn hóa lại.

    Args:
        candidate_skills: Kỹ năng của ứng viên.
        domain_skills: Kỹ năng đặc thù lĩnh vực hay JD.
        candidate_job_titles: Các chức danh trong lịch sử ứng viên.
        target_job_titles: Chức danh mục tiêu của JD.
        embedding_engine: Engine embedding cho cosine chức danh.
        skill_weight: Trọng số Jaccard kỹ năng.
        title_weight: Trọng số cosine chức danh.
        esco_kb: Knowledge base ESCO tùy chọn để chuẩn hóa nhãn kỹ năng.

    Returns:
        Điểm lĩnh vực trong khoảng từ 0.0 đến 1.0.

    Raises:
        ValueError: Nếu trọng số âm hoặc tổng trọng số khả dụng bằng 0.
    """
    if skill_weight < 0.0 or title_weight < 0.0:
        raise ValueError("Trọng số domain không được âm.")

    alias_map = _build_esco_alias_map(esco_kb)
    candidate_skill_set = _canonical_skill_set(candidate_skills, alias_map)
    domain_skill_set = _canonical_skill_set(domain_skills, alias_map)
    candidate_titles = _as_text_list(candidate_job_titles)
    target_titles = _as_text_list(target_job_titles)

    components: List[float] = []
    weights: List[float] = []
    if domain_skill_set:
        union = candidate_skill_set.union(domain_skill_set)
        jaccard = (
            len(candidate_skill_set.intersection(domain_skill_set)) / len(union)
            if union
            else 0.0
        )
        components.append(_clamp_score(jaccard))
        weights.append(skill_weight)
    if target_titles:
        components.append(
            _title_similarity(candidate_titles, target_titles, embedding_engine)
        )
        weights.append(title_weight)

    if not components:
        return 0.0
    weight_total = sum(weights)
    if weight_total <= 0.0:
        raise ValueError("Tổng trọng số domain phải lớn hơn 0.")
    return _clamp_score(
        sum(component * weight for component, weight in zip(components, weights))
        / weight_total
    )


__all__: List[str] = [
    "DEGREE_RANKING",
    "domain_score",
    "edu_score",
    "exp_score",
    "skill_score",
]
