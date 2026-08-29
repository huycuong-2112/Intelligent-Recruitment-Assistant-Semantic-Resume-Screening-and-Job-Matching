def group_by_category(features: list) -> dict:
    grouped = {}
    for feature in features or []:
        if isinstance(feature, dict):
            grouped.setdefault(feature.get("category", "Other"), []).append(feature.get("name", ""))
    return grouped
