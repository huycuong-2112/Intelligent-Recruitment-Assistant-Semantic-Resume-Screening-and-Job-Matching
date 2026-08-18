import hashlib


def generate_mock_scores(seed_text: str) -> dict:
    """
    Sinh điểm số giả (deterministic) cho 4 tiêu chí dựa trên tên file CV.
    Dùng để test giao diện Candidate Ranking khi model thật chưa được tích hợp.
    Cùng 1 tên file sẽ luôn cho ra cùng 1 bộ điểm (không random lung tung mỗi lần rerun).
    """
    aspects = ["Edu Score", "Skill Score", "Domain Score", "Exp Score"]
    scores = {}
    for aspect in aspects:
        h = hashlib.md5(f"{seed_text}_{aspect}".encode()).hexdigest()
        value = int(h[:4], 16) / 0xFFFF  # chuẩn hóa về khoảng 0-1
        scores[aspect] = round(0.4 + value * 0.55, 3)  # giữ điểm trong khoảng hợp lý 0.4 - 0.95
    return scores