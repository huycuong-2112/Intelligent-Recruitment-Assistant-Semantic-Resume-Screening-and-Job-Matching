import hashlib

CATEGORIES = ["Edu", "Skill", "Domain", "Exp"]

MOCK_FEATURE_POOL = {
    "Edu": ["Đại học Bách Khoa", "Thạc sĩ CNTT", "Cử nhân Kinh tế", "Chứng chỉ AWS"],
    "Skill": ["Python", "SQL", "Machine Learning", "FastAPI", "Docker", "Git"],
    "Domain": ["Ngành Fintech", "Ngành E-commerce", "Ngành Y tế", "Ngành Giáo dục"],
    "Exp": ["3 năm kinh nghiệm", "Quản lý dự án", "Từng làm Senior", "Thực tập 6 tháng"],
}


def _hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:4], 16)


def mock_extract_features(seed_files: list) -> list:
    """
    Trả về danh sách feature dạng dict: {"name": ..., "category": ...}
    Deterministic theo tên file — mỗi trường (Edu/Skill/Domain/Exp) sẽ có 1-2 feature.
    """
    seed = "_".join(sorted(seed_files))
    results = []
    for category, pool in MOCK_FEATURE_POOL.items():
        count = 1 + (_hash_int(seed + category) % 2)
        chosen_positions = set()
        i = 0
        while len(chosen_positions) < count and i < 10:
            pos = _hash_int(f"{seed}_{category}_{i}") % len(pool)
            chosen_positions.add(pos)
            i += 1
        for pos in chosen_positions:
            results.append({"name": pool[pos], "category": category})
    return results


def group_by_category(features: list) -> dict:
    """Gom danh sách feature (dict có 'category') thành dict {category: [tên feature,...]}."""
    grouped = {cat: [] for cat in CATEGORIES}
    for feat in features:
        grouped[feat["category"]].append(feat["name"])
    return grouped