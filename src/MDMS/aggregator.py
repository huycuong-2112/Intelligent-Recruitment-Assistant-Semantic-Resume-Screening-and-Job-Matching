"""Bộ tổng hợp điểm Multi-Dimensional Matching Score (MDMS)."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Union


ScoreMapping = Mapping[str, Any]
ScoreInput = Union[Real, ScoreMapping, None]


class MDMSAggregator:
    """Tổng hợp bốn điểm thành phần MDMS theo bộ trọng số ngành nghề.

    Các thành phần chuẩn của MDMS là ``skill``, ``experience``, ``education`` và
    ``domain``. Điểm đầu vào có thể ở thang ``[0, 1]`` hoặc ``[0, 100]``; điểm
    kết quả luôn trả đồng thời hai thang để phục vụ API lẫn giao diện.
    """

    DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
        "IT": {
            "skill": 0.50,
            "experience": 0.15,
            "education": 0.15,
            "domain": 0.20,
        },
        "Finance": {
            "skill": 0.25,
            "experience": 0.20,
            "education": 0.30,
            "domain": 0.25,
        },
        "Healthcare": {
            "skill": 0.20,
            "experience": 0.25,
            "education": 0.35,
            "domain": 0.20,
        },
        "General": {
            "skill": 0.30,
            "experience": 0.25,
            "education": 0.20,
            "domain": 0.25,
        },
    }
    """Bộ trọng số mặc định theo ngành, mỗi bộ có tổng chính xác bằng 1.0."""

    _COMPONENT_KEYS: tuple[str, ...] = (
        "skill",
        "experience",
        "education",
        "domain",
    )
    _INDUSTRY_ALIASES: Dict[str, str] = {
        "it": "IT",
        "information technology": "IT",
        "technology": "IT",
        "tech": "IT",
        "finance": "Finance",
        "financial services": "Finance",
        "healthcare": "Healthcare",
        "health care": "Healthcare",
        "medical": "Healthcare",
        "general": "General",
    }

    def __init__(
        self,
        industry: str = "General",
        weights: Optional[Mapping[str, Real]] = None,
    ) -> None:
        """Khởi tạo aggregator với một ngành hoặc bộ trọng số tùy chỉnh.

        Args:
            industry: Tên ngành áp dụng. Tên không nhận diện được dùng bộ General.
            weights: Bộ trọng số bốn thành phần tùy chỉnh cho phiên làm việc này.

        Raises:
            ValueError: Nếu bộ trọng số tùy chỉnh không hợp lệ.
        """
        self.industry = self._normalise_industry(industry)
        selected_weights: Mapping[str, Real]
        if weights is None:
            selected_weights = self.DEFAULT_WEIGHTS[self.industry]
        else:
            selected_weights = weights
        self._validate_weights(selected_weights)
        self.weights: Dict[str, float] = {
            component: float(selected_weights[component])
            for component in self._COMPONENT_KEYS
        }

    @classmethod
    def _normalise_industry(cls, industry: Optional[Any]) -> str:
        """Chuẩn hóa tên ngành về khóa có trong ``DEFAULT_WEIGHTS``.

        Args:
            industry: Tên ngành đầu vào.

        Returns:
            Một trong bốn khóa ngành mặc định; dùng ``General`` khi không rõ.
        """
        candidate = str(industry).strip().casefold() if industry is not None else ""
        return cls._INDUSTRY_ALIASES.get(candidate, "General")

    @classmethod
    def _validate_weights(cls, weights: Mapping[str, Real]) -> None:
        """Kiểm tra bộ trọng số MDMS hợp lệ.

        Bộ trọng số phải có đúng bốn thành phần chuẩn, không âm, hữu hạn và có
        tổng bằng 1.0 trong sai số dấu chấm động nhỏ.

        Args:
            weights: Mapping trọng số cần xác thực.

        Raises:
            ValueError: Nếu thiếu/thừa thành phần, có trọng số âm, không hữu hạn
                hoặc tổng khác 1.0.
        """
        if not isinstance(weights, Mapping):
            raise ValueError("Weights phải là mapping gồm bốn thành phần MDMS.")

        provided_keys = set(weights.keys())
        expected_keys = set(cls._COMPONENT_KEYS)
        if provided_keys != expected_keys:
            missing = sorted(expected_keys.difference(provided_keys))
            unexpected = sorted(provided_keys.difference(expected_keys))
            raise ValueError(
                "Weights phải có đúng các khóa skill, experience, education, "
                f"domain; thiếu={missing}, thừa={unexpected}."
            )

        numeric_weights: Dict[str, float] = {}
        for component in cls._COMPONENT_KEYS:
            value = weights[component]
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"Trọng số '{component}' phải là số thực.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(
                    f"Trọng số '{component}' phải là số hữu hạn không âm."
                )
            numeric_weights[component] = numeric_value

        if not math.isclose(sum(numeric_weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Tổng các trọng số MDMS phải bằng 1.0.")

    @staticmethod
    def _normalise_component_score(value: Optional[Any]) -> float:
        """Chuẩn hóa điểm thành phần ở thang 0-1 hoặc 0-100 về thang 0-1.

        Args:
            value: Điểm đầu vào có thể là số, chuỗi số hoặc ``None``.

        Returns:
            Điểm đã chặn trong khoảng từ 0.0 đến 1.0. Giá trị lỗi nhận 0.0.
        """
        if value is None or isinstance(value, bool):
            return 0.0
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(numeric_value):
            return 0.0
        if numeric_value > 1.0:
            numeric_value /= 100.0
        return min(1.0, max(0.0, numeric_value))

    @staticmethod
    def _lookup_score(scores: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
        """Lấy điểm đầu tiên có mặt theo danh sách bí danh không phân biệt hoa thường.

        Args:
            scores: Dictionary điểm thành phần.
            aliases: Các tên khóa có thể dùng cho cùng một thành phần.

        Returns:
            Giá trị điểm thô, hoặc ``None`` khi không tìm thấy.
        """
        direct_keys = {alias.casefold() for alias in aliases}
        for key, value in scores.items():
            if str(key).strip().casefold() in direct_keys:
                return value
        return None

    @classmethod
    def _coerce_scores(
        cls,
        skill_score: ScoreInput,
        exp_score: Optional[Any],
        edu_score: Optional[Any],
        domain_score: Optional[Any],
        scores: Optional[ScoreMapping],
    ) -> Dict[str, float]:
        """Chuẩn hóa các kiểu đầu vào điểm được hỗ trợ thành bốn thành phần MDMS.

        Args:
            skill_score: Điểm skill riêng lẻ hoặc mapping tất cả điểm.
            exp_score: Điểm experience riêng lẻ.
            edu_score: Điểm education riêng lẻ.
            domain_score: Điểm domain riêng lẻ.
            scores: Mapping tất cả điểm, ưu tiên hơn mapping ở ``skill_score``.

        Returns:
            Dictionary bốn điểm thành phần ở thang ``[0, 1]``.
        """
        mapping: Mapping[str, Any] = {}
        if isinstance(skill_score, Mapping):
            mapping = skill_score
        if scores is not None:
            mapping = scores

        if mapping:
            raw_values = {
                "skill": cls._lookup_score(
                    mapping,
                    ("skill", "skills", "skill_score", "skill score"),
                ),
                "experience": cls._lookup_score(
                    mapping,
                    (
                        "experience",
                        "exp",
                        "experience_score",
                        "exp_score",
                        "experience score",
                        "exp score",
                    ),
                ),
                "education": cls._lookup_score(
                    mapping,
                    ("education", "edu", "education_score", "edu_score", "edu score"),
                ),
                "domain": cls._lookup_score(
                    mapping,
                    ("domain", "domain_score", "domain score"),
                ),
            }
        else:
            raw_values = {
                "skill": skill_score,
                "experience": exp_score,
                "education": edu_score,
                "domain": domain_score,
            }
        return {
            component: cls._normalise_component_score(raw_values[component])
            for component in cls._COMPONENT_KEYS
        }

    def compute_total_score(
        self,
        skill_score: ScoreInput = None,
        exp_score: Optional[Any] = None,
        edu_score: Optional[Any] = None,
        domain_score: Optional[Any] = None,
        industry: Optional[str] = None,
        scores: Optional[ScoreMapping] = None,
    ) -> Dict[str, Any]:
        """Tính điểm MDMS tổng và trả phân rã chi tiết.

        Có thể truyền bốn điểm riêng lẻ theo thứ tự ``skill, experience,
        education, domain`` hoặc truyền mapping qua đối số ``skill_score`` hay
        ``scores``. Khi truyền ``industry``, bộ trọng số mặc định của ngành đó
        được dùng cho riêng lần tính này; bộ trọng số tùy chỉnh của instance vẫn
        được dùng nếu không truyền ``industry``.

        Args:
            skill_score: Điểm skill hoặc mapping chứa bốn điểm thành phần.
            exp_score: Điểm kinh nghiệm.
            edu_score: Điểm học vấn.
            domain_score: Điểm lĩnh vực.
            industry: Ngành cần áp dụng trọng số mặc định cho lần tính này.
            scores: Mapping thay thế, ưu tiên cao nhất khi được cung cấp.

        Returns:
            Dictionary gồm component scores, weighted scores, tổng ở thang 0-1
            và tổng ở thang 0-100.
        """
        component_scores = self._coerce_scores(
            skill_score,
            exp_score,
            edu_score,
            domain_score,
            scores,
        )
        if industry is None:
            applied_industry = self.industry
            applied_weights = self.weights
        else:
            applied_industry = self._normalise_industry(industry)
            applied_weights = self.DEFAULT_WEIGHTS[applied_industry]

        weighted_scores = {
            component: component_scores[component] * applied_weights[component]
            for component in self._COMPONENT_KEYS
        }
        total_score_0_1 = min(1.0, max(0.0, sum(weighted_scores.values())))
        total_score_0_100 = total_score_0_1 * 100.0
        breakdown = {
            component: {
                "score": component_scores[component],
                "weight": applied_weights[component],
                "weighted_score": weighted_scores[component],
            }
            for component in self._COMPONENT_KEYS
        }

        return {
            "industry": applied_industry,
            "weights": dict(applied_weights),
            "component_scores": component_scores,
            "weighted_scores": weighted_scores,
            "breakdown": breakdown,
            "total_score_0_1": total_score_0_1,
            "total_score_0_100": total_score_0_100,
        }


DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = MDMSAggregator.DEFAULT_WEIGHTS
"""Bí danh cấp mô-đun cho bộ trọng số mặc định của ``MDMSAggregator``."""


__all__ = ["DEFAULT_WEIGHTS", "MDMSAggregator"]
