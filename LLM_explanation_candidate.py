from __future__ import annotations

import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT & GROQ API SETUP
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq, RateLimitError
except ImportError:
    Groq = None
    RateLimitError = Exception

# Compatibility default: credentials are supplied only through the environment.
HARDCODED_GROQ_API_KEY = ""

# Không hardcode key - Ưu tiên đọc từ biến môi trường
# HARDCODED_GROQ_API_KEY = ""

def get_groq_client_cycler():
    raw_keys = (
        os.getenv("GROQ_API_KEYS")
        or os.getenv("GROQ_API_KEY")
        or HARDCODED_GROQ_API_KEY
        or ""
    )
    keys = [k.strip() for k in raw_keys.split(",") if k.strip().startswith("gsk_")]
    if not keys or not Groq:
        return None
    return itertools.cycle([Groq(api_key=k) for k in keys])

def get_active_groq_models(client) -> List[str]:
    try:
        models_data = client.models.list().data
        chat_models = [
            m.id for m in models_data 
            if "whisper" not in m.id and "guard" not in m.id and "embedding" not in m.id
        ]
        if chat_models:
            return chat_models
    except Exception:
        pass
    return GROQ_MODELS

GROQ_MODELS = [
    "gemma2-9b-it",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview",
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# ---------------------------------------------------------------------------
# 2. PATH CONFIGURATION (Tự động thích ứng vị trí file)
# ---------------------------------------------------------------------------
current_file = Path(__file__).resolve()
# Nếu chạy ở root hay thư mục con đều tự nhận diện
project_root = current_file.parent.parent

POTENTIAL_INPUT_PATHS = [
    # project_root / "Data" / "Results" / "IT" / "XAI" / "Examples" / "cv_024_xai.json",
    project_root / "Data" / "Results" / "IT" / "XAI" / "Examples" / "jd_001_cv_example_xai.json",
]

DEFAULT_OUTPUT_JSON = current_file.parent / "candidate_feedback.json"
DEFAULT_OUTPUT_MD = current_file.parent / "candidate_feedback.md"

# ---------------------------------------------------------------------------
# 3. SCHEMA DEFINITIONS (UNIVERSAL FEEDBACK V1)
# ---------------------------------------------------------------------------
class CVImprovementItem(BaseModel):
    priority: str = Field(..., description="Tiêu đề ưu tiên")
    guidance: str = Field(..., description="Hướng dẫn chi tiết")
    example: Optional[str] = Field(default="", description="Ví dụ minh họa cụ thể")

class CandidateFeedbackNarrative(BaseModel):
    fit_level: str = Field(..., description="Mức độ phù hợp: Thấp, Trung bình, Khá, Cao")
    summary: str = Field(..., description="Tóm tắt nhận định tổng quan")
    primary_factor: str = Field(..., description="Yếu tố ảnh hưởng lớn nhất đến điểm")
    strengths: List[str] = Field(default_factory=list, description="Danh sách điểm mạnh")
    gaps: List[str] = Field(default_factory=list, description="Danh sách khoảng cách với JD")
    cv_improvements: List[CVImprovementItem] = Field(default_factory=list, description="Các ưu tiên nâng cấp CV")

class CandidateFeedbackSchema(BaseModel):
    schema_version: str = "candidate_feedback_v1"
    cv_id: str
    jd_id: str
    job_title: str
    mdms_score: float
    mdms_percentage: str
    feedback: CandidateFeedbackNarrative

# ---------------------------------------------------------------------------
# 4. UNIVERSAL COMPACT PAYLOAD BUILDER
# ---------------------------------------------------------------------------
class CompactPayloadBuilder:
    @staticmethod
    def extract_skills(skill_dim: Dict[str, Any]) -> tuple[List[str], List[str]]:
        if "requirements" in skill_dim:
            reqs = skill_dim.get("requirements", [])
            matched = [r.get("jd_skill") for r in reqs if r.get("status") == "matched"]
            missing = [
                r.get("jd_skill") for r in reqs 
                if r.get("status") == "no_evidence" and r.get("importance") == "required"
            ]
            return matched, missing
        
        matched = skill_dim.get("matched_required", [])
        missing = skill_dim.get("missing_required", [])
        return matched, missing

    @staticmethod
    def extract_experience(exp_dim: Dict[str, Any]) -> tuple[float, float]:
        years_info = exp_dim.get("years", {})
        details = years_info.get("details") or {}
        cand_years = details.get("candidate_years") if "candidate_years" in details else years_info.get("candidate_years", 0.0)
        min_years = details.get("minimum_years") if "minimum_years" in details else years_info.get("minimum_years", 0.0)
        return float(cand_years or 0.0), float(min_years or 0.0)

    @staticmethod
    def extract_education(edu_dim: Dict[str, Any]) -> tuple[str, str, bool]:
        degree_info = edu_dim.get("degree", {})
        field_info = edu_dim.get("field", {})

        deg_details = degree_info.get("details") or {}
        field_details = field_info.get("details") or {}

        deg = deg_details.get("candidate") or degree_info.get("candidate") or "Bachelor"
        field = field_details.get("candidate") or field_info.get("candidate") or "Computer Science"

        deg_status = str(degree_info.get("status", "")).lower()
        field_status = str(field_info.get("status", "")).lower()
        is_matched = deg_status in ("satisfied", "matched") and field_status in ("matched",)

        return deg, field, is_matched

    @classmethod
    def build(cls, xai_data: Dict[str, Any]) -> Dict[str, Any]:
        cv_id = xai_data.get("cv_id") or (xai_data.get("metadata") or {}).get("cv_id", "CV")
        jd_id = xai_data.get("jd_id") or (xai_data.get("metadata") or {}).get("jd_id", "JD")
        job_title = xai_data.get("job_title") or (xai_data.get("metadata") or {}).get("job_title", "Vị trí tuyển dụng")

        dims = xai_data.get("dimensions", {})
        score = xai_data.get("decision", {}).get("final_score", 0.0)

        matched_skills, missing_skills = cls.extract_skills(dims.get("skill", {}))
        cand_years, min_years = cls.extract_experience(dims.get("experience", {}))
        degree, field, edu_matched = cls.extract_education(dims.get("education", {}))

        return {
            "cv_id": cv_id,
            "jd_id": jd_id,
            "job_title": job_title,
            "score": score,
            "score_pct": f"{score * 100:.1f}%",
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "candidate_years": cand_years,
            "minimum_years": min_years,
            "degree": degree,
            "field": field,
            "education_matched": edu_matched
        }

# ---------------------------------------------------------------------------
# 5. GROQ PRODUCTION PROMPT & LLM CALL
# ---------------------------------------------------------------------------
GROQ_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn tuyển dụng và tối ưu CV kỹ thuật cho mọi ngành nghề.
Dựa vào JSON facts từ hệ thống đánh giá MDMS, hãy lập Báo cáo Phản hồi Hồ sơ cho ứng viên bằng tiếng Việt theo phong cách xây dựng, khách quan và bám sát bằng chứng.

QUY TẮC PHÂN TÍCH:
1. Nhận diện lĩnh vực chuyên môn từ `job_title` và nhóm kỹ năng.
2. Mức độ phù hợp: "Thấp" (score < 0.40), "Trung bình" (0.40 - 0.69), "Cao" (>= 0.70).
3. Điểm mạnh: Ghi nhận bằng cấp, số năm kinh nghiệm (nếu có) và kỹ năng đã khớp.
4. Khoảng cách: Nêu rõ những kỹ năng bắt buộc chưa tìm thấy bằng chứng theo cú pháp: "Bạn đã [hành động chuyên môn] trong thực tế."
5. Nâng cấp CV:
   - Ưu tiên 1: Đề xuất dự án mẫu thực chiến áp dụng trực tiếp các kỹ năng đang thiếu (`missing_skills`).
   - Ưu tiên 2: Hướng dẫn ứng viên làm rõ vai trò kỹ thuật trực tiếp và số liệu định lượng trong phần Experience.
   - Ưu tiên 3: Gợi ý xây dựng portfolio, repo GitHub, demo kiểm chứng.

CẤU TRÚC JSON BẮT BUỘC:
{
  "fit_level": "Thấp" | "Trung bình" | "Cao",
  "summary": "Đoạn văn 2-3 câu nhận xét tổng quan.",
  "primary_factor": "Yếu tố chính ảnh hưởng đến điểm.",
  "strengths": ["..."],
  "gaps": ["Bạn đã..."],
  "cv_improvements": [
    {
      "priority": "Ưu tiên 1 — Bổ sung evidence kỹ thuật",
      "guidance": "...",
      "example": "..."
    },
    {
      "priority": "Ưu tiên 2 — Làm rõ phần Experience",
      "guidance": "...",
      "example": ""
    },
    {
      "priority": "Ưu tiên 3 — GitHub / Portfolio",
      "guidance": "...",
      "example": ""
    }
  ]
}
"""

def call_groq_explanation(compact_payload: Dict[str, Any], key_cycler: Any) -> Dict[str, Any]:
    user_prompt = "Respond with a valid JSON object only.\nINPUT FACTS:\n" + json.dumps(compact_payload, ensure_ascii=False, indent=2)
    last_error = None

    for attempt in range(1, 4):
        client = next(key_cycler)
        available_models = get_active_groq_models(client)
        for model_name in available_models:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                raw_json = response.choices[0].message.content
                if raw_json:
                    return json.loads(raw_json)
            except RateLimitError as rle:
                last_error = rle
                time.sleep(3.0)
                break
            except Exception as exc:
                last_error = exc
                continue

    raise last_error or RuntimeError("Groq explanation call failed across all models.")

# ---------------------------------------------------------------------------
# 6. UNIVERSAL OFFLINE GENERATOR (Xử lý mượt văn phong theo số năm)
# ---------------------------------------------------------------------------
def generate_universal_offline_explanation(compact_payload: Dict[str, Any]) -> Dict[str, Any]:
    job_title = compact_payload.get("job_title", "Vị trí tuyển dụng")
    score = compact_payload.get("score", 0.0)
    matched_skills = compact_payload.get("matched_skills", [])
    missing_skills = compact_payload.get("missing_skills", [])
    cand_years = compact_payload.get("candidate_years", 0.0)
    min_years = compact_payload.get("minimum_years", 0.0)
    degree = compact_payload.get("degree", "Bằng cấp chuyên ngành")
    field = compact_payload.get("field", "Chuyên ngành liên quan")

    # Đoạn mô tả kinh nghiệm thích ứng văn phong
    if cand_years > 0:
        exp_phrase = f"và có thời gian làm việc tích lũy ({cand_years} năm)"
    else:
        exp_phrase = "mặc dù chưa ghi nhận số năm kinh nghiệm làm việc chính thức"

    if score >= 0.70:
        fit_level = "Cao"
        summary = (
            f"Hồ sơ ứng viên thể hiện mức độ tương thích tốt với vị trí {job_title}. "
            f"Ứng viên có bằng cấp phù hợp ({degree} — {field}), {exp_phrase} và "
            f"đáp ứng đầy đủ bằng chứng cho phần lớn các kỹ năng trọng yếu được yêu cầu."
        )
        primary_factor = "sự đồng bộ cao giữa kinh nghiệm thực tế, kỹ năng chuyên môn và yêu cầu tuyển dụng."
    elif score >= 0.40:
        fit_level = "Trung bình"
        missing_preview = ", ".join(missing_skills[:3]) if missing_skills else "một số yêu cầu kỹ thuật"
        summary = (
            f"Hồ sơ đáp ứng tiêu chuẩn về bằng cấp ({degree} — {field}) {exp_phrase} đối với vị trí {job_title}. "
            f"Tuy nhiên, mức độ phù hợp ở mức trung bình do CV còn thiếu bằng chứng rõ ràng cho một số kỹ năng cốt lõi như {missing_preview}."
        )
        primary_factor = f"còn thiếu bằng chứng về một số kỹ năng chuyên môn cốt lõi ({missing_preview})."
    else:
        fit_level = "Thấp"
        missing_preview = ", ".join(missing_skills[:4]) if missing_skills else "các kỹ năng cốt lõi"
        summary = (
            f"Hồ sơ đáp ứng yêu cầu về bằng cấp ({degree} — {field}) {exp_phrase}. "
            f"Tuy nhiên, mức độ phù hợp với vị trí {job_title} hiện còn thấp vì CV chưa cung cấp đủ bằng chứng về "
            f"{missing_preview} — đây là nhóm yêu cầu kỹ thuật cốt lõi của JD."
        )
        primary_factor = f"thiếu bằng chứng thực hiện dự án hoặc kỹ năng chuyên môn trực tiếp liên quan đến {job_title}."

    strengths = []
    if degree and field:
        strengths.append(f"Bằng cấp phù hợp với nền tảng {field} ({degree}).")
    if cand_years > 0:
        strengths.append(f"Có {cand_years} năm kinh nghiệm làm việc, cho thấy exposure với môi trường kỹ thuật.")
    if matched_skills:
        strengths.append(f"Đã chứng minh năng lực thực tế với các kỹ năng: {', '.join(matched_skills)}.")
    else:
        strengths.append(f"Kinh nghiệm và kiến thức hiện tại có thể là nền tảng để chuyển sang vị trí {job_title}.")

    gaps = []
    for skill in missing_skills[:5]:
        gaps.append(f"Bạn đã áp dụng hoặc làm chủ {skill} trong dự án thực tế.")
    if cand_years < min_years:
        gaps.append(f"Thời gian kinh nghiệm tích lũy ({cand_years} năm) chưa đạt mốc tối thiểu yêu cầu ({min_years} năm).")
    if not gaps:
        gaps.append("Chưa thể hiện rõ các dự án quy mô lớn hoặc kỹ thuật chuyên sâu đặc thù.")

    top_missing_str = ", ".join(missing_skills[:3]) if missing_skills else "kỹ năng yêu cầu"
    example_project_title = f"Dự án thực tế ứng dụng {top_missing_str}" if missing_skills else f"Dự án chuyên môn {job_title}"
    
    cv_improvements = [
        {
            "priority": "Ưu tiên 1 — Bổ sung evidence kỹ thuật",
            "guidance": f"Thay vì chỉ thêm:\n\n\"{top_missing_str}\"\n\nhãy thêm project cụ thể:",
            "example": (
                f"{example_project_title} — {top_missing_str}\n\n"
                f"Xây dựng và hoàn thiện module/dự án áp dụng {top_missing_str}; mô tả rõ ràng khâu tiền xử lý, thuật toán, công nghệ sử dụng và kết quả đo lường cụ thể. Quản lý mã nguồn bằng hệ thống quản trị phiên bản."
            )
        },
        {
            "priority": "Ưu tiên 2 — Làm rõ phần Experience",
            "guidance": "Nếu bạn thực sự từng code, xử lý data hoặc xây dựng automation trong công việc hiện tại, hãy mô tả bạn trực tiếp xây dựng gì, sử dụng công nghệ nào và kết quả ra sao.",
            "example": ""
        },
        {
            "priority": "Ưu tiên 3 — GitHub / Portfolio",
            "guidance": f"Thêm GitHub hoặc portfolio chứa các project có source code liên quan đến {top_missing_str}. Đây là evidence mạnh hơn một danh sách Skills đơn thuần.",
            "example": ""
        }
    ]

    return {
        "fit_level": fit_level,
        "summary": summary,
        "primary_factor": primary_factor,
        "strengths": strengths,
        "gaps": gaps,
        "cv_improvements": cv_improvements
    }

# ---------------------------------------------------------------------------
# 7. UNIVERSAL BREAKDOWN TABLE & MARKDOWN FORMATTER
# ---------------------------------------------------------------------------
def build_universal_breakdown_table(xai_data: Dict[str, Any]) -> str:
    dims = xai_data.get("dimensions", {})
    registry = xai_data.get("evidence_registry", {})
    rows = []

    # 1. Trích xuất Kỹ năng (Hỗ trợ 2 chiều Format)
    skill_dim = dims.get("skill", {})
    if "requirements" in skill_dim:
        for req in skill_dim.get("requirements", []):
            skill_name = req.get("jd_skill", "Kỹ năng")
            importance = req.get("importance", "required")
            status = req.get("status", "no_evidence")
            matched_cv_skill = req.get("matched_cv_skill")
            
            display_name = skill_name if importance == "required" else f"{skill_name} *(Ưu tiên)*"
            if status == "matched":
                ev_str = f"Đã xác thực trong CV" + (f" (khớp: `{matched_cv_skill}`)" if matched_cv_skill else "")
                rows.append(f"| {display_name} | ✅ Đạt | {ev_str} |")
            else:
                rows.append(f"| {display_name} | ❌ Chưa xác nhận | Không tìm thấy bằng chứng/project cụ thể trong CV |")
    else:
        for s in skill_dim.get("matched_required", []):
            rows.append(f"| {s} | ✅ Đạt | Có bằng chứng ghi nhận trong CV |")
        for s in skill_dim.get("missing_required", []):
            rows.append(f"| {s} | ❌ Chưa xác nhận | Không thấy đề cập trong Skills/Projects/Experience |")

    # 2. Trích xuất Kinh nghiệm & Tìm Snippet mạnh nhất
    exp_dim = dims.get("experience", {})
    cand_years, min_years = CompactPayloadBuilder.extract_experience(exp_dim)
    exp_score = exp_dim.get("score", 0.0)

    best_snippet = ""
    evidence_block = exp_dim.get("evidence", {})
    if "responsibilities" in evidence_block:
        resps = evidence_block.get("responsibilities", [])
        if resps:
            # Chọn responsibility có match score cao nhất
            best_resp = max(resps, key=lambda r: r.get("score", 0.0))
            best_snippet = best_resp.get("matched_text", "").strip()
    elif "selected_evidence" in exp_dim:
        for sel in exp_dim.get("selected_evidence", []):
            ref_id = sel.get("evidence_ref")
            if ref_id in registry:
                best_snippet = registry[ref_id].get("source_text", "").strip()
                break

    if exp_score >= 0.7:
        exp_status = "✅ Đạt"
        exp_desc = f"{cand_years} năm kinh nghiệm đáp ứng tốt yêu cầu ({min_years}+ năm)"
    elif exp_score >= 0.3:
        exp_status = "🟡 Một phần"
        short_snip = (best_snippet[:65] + "...") if len(best_snippet) > 65 else best_snippet
        if cand_years > 0:
            exp_desc = f"{cand_years} năm kinh nghiệm ({short_snip}), nhưng chưa thể hiện đủ AI implementation thực chiến"
        else:
            exp_desc = f"Chưa có năm kinh nghiệm chính thức (dự án: {short_snip}), cần bổ sung thêm bằng chứng AI thực chiến"
    else:
        exp_status = "❌ Chưa đạt"
        exp_desc = f"{cand_years} năm kinh nghiệm (chưa đạt yêu cầu tối thiểu {min_years} năm)"

    rows.append(f"| Relevant experience | {exp_status} | {exp_desc} |")

    # 3. Trích xuất Học vấn
    deg_cand, field_cand, is_matched = CompactPayloadBuilder.extract_education(dims.get("education", {}))
    edu_dim = dims.get("education", {})
    degree_info = edu_dim.get("degree", {})
    deg_status = str(degree_info.get("status", "")).lower()

    if is_matched:
        edu_status = "✅ Đạt"
    elif deg_status in ("satisfied", "matched") or edu_dim.get("score", 0.0) >= 0.5:
        edu_status = "🟡 Một phần"
    else:
        edu_status = "❌ Chưa đạt"

    rows.append(f"| Education | {edu_status} | {deg_cand} in {field_cand} |")

    return "| Tiêu chí | Mức độ đáp ứng | Evidence từ CV |\n|---|---|---|\n" + "\n".join(rows)

def format_candidate_feedback_report(xai_data: Dict[str, Any], feedback_data: Dict[str, Any]) -> str:
    job_title = xai_data.get("job_title") or (xai_data.get("metadata") or {}).get("job_title", "Vị trí tuyển dụng")
    jd_id = xai_data.get("jd_id") or (xai_data.get("metadata") or {}).get("jd_id", "jd_000")
    score_pct = xai_data.get("decision", {}).get("final_score", 0.0) * 100

    breakdown_table = build_universal_breakdown_table(xai_data)
    strengths_items = "\n\n".join(feedback_data.get("strengths", []))
    gaps_items = "\n\n".join(feedback_data.get("gaps", []))

    improv_sections = []
    for imp in feedback_data.get("cv_improvements", []):
        p_title = imp.get("priority", "")
        p_guide = imp.get("guidance", "")
        p_ex = imp.get("example", "")
        block = f"{p_title}\n{p_guide}"
        if p_ex:
            block += f"\n\n{p_ex}\n\nĐiều này giúp hệ thống có evidence thực tế, thay vì chỉ nhận diện skill keyword."
        improv_sections.append(block)

    improvements_text = "\n\n".join(improv_sections)

    # Đọc kỹ năng thiếu chuẩn Adapter (hoạt động cho cả 2 format)
    _, missing_skills = CompactPayloadBuilder.extract_skills(xai_data.get("dimensions", {}).get("skill", {}))
    if missing_skills:
        missing_skills_sample = ", ".join(missing_skills[:3])
        note_skills_phrase = f"Nếu bạn có {missing_skills_sample} nhưng CV chưa thể hiện"
    else:
        note_skills_phrase = "Nếu bạn có các kỹ năng cốt lõi nhưng CV chưa thể hiện rõ"

    return f"""Báo cáo Phản hồi Hồ sơ Ứng viên
Vị trí: {job_title} ({jd_id})
MDMS: {score_pct:.1f}%

1. Kết luận
Mức độ phù hợp: {feedback_data.get('fit_level', 'Thấp')}
{feedback_data.get('summary', '')}
Yếu tố ảnh hưởng lớn nhất đến điểm: {feedback_data.get('primary_factor', '')}

2. MDMS Breakdown
{breakdown_table}

3. Những gì CV đang chứng minh
Điểm mạnh

{strengths_items}

Khoảng cách với JD
CV hiện chưa chứng minh được:

{gaps_items}

4. Nếu muốn nang cap CV
{improvements_text}

5. Điều cần lưu ý
MDMS đánh giá dựa trên bằng chứng được thể hiện trong CV, không phải toàn bộ kỹ năng thực tế của ứng viên.
{note_skills_phrase}, điểm hiện tại có thể thấp hơn năng lực thực tế của bạn."""

# ---------------------------------------------------------------------------
# 8. RESOLVE INPUT FILE / DIRECTORY
# ---------------------------------------------------------------------------
def find_input_source() -> Path:
    for candidate in POTENTIAL_INPUT_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Không tìm thấy file XAI input JSON phù hợp trong hệ thống.")

# ---------------------------------------------------------------------------
# 9. MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PIPELINE TẠO BÁO CÁO PHẢN HỒI HỒ SƠ ỨNG VIÊN (UNIVERSAL DUAL-FORMAT ENGINE)")
    print(f"Project Root   : {project_root}")

    input_path = find_input_source()
    output_json = DEFAULT_OUTPUT_JSON
    output_md = DEFAULT_OUTPUT_MD
    output_json.parent.mkdir(parents=True, exist_ok=True)

    key_cycler = get_groq_client_cycler()
    mode = "ONLINE (Groq Cloud LLM)" if key_cycler else "OFFLINE (Universal Fallback Engine)"

    print(f"Chế độ chạy    : {mode}")
    print(f"Nguồn XAI Input: {input_path}")
    print(f"File JSON Đích : {output_json}")
    print(f"File MD Đích   : {output_md}")
    print("=" * 80)

    items: List[Dict[str, Any]] = []
    if input_path.is_file():
        content = json.loads(input_path.read_text(encoding="utf-8"))
        items = content if isinstance(content, list) else [content]
    elif input_path.is_dir():
        for f in sorted(input_path.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    items.extend(data)
                elif isinstance(data, dict):
                    items.append(data)
            except Exception:
                continue

    if not items:
        print(f"⚠️ Không tìm thấy bản ghi XAI hợp lệ trong: {input_path}")
        return

    print(f"📂 Tìm thấy {len(items)} hồ sơ matching cần xử lý.\n")

    final_results: List[Dict[str, Any]] = []
    markdown_reports: List[str] = []

    for idx, xai in enumerate(items, 1):
        cv_id = xai.get("cv_id", f"cv_{idx:03d}")
        jd_id = xai.get("jd_id", "jd_001")
        score = xai.get("decision", {}).get("final_score", 0.0)

        print(f"[{idx}/{len(items)}] Đang xử lý phản hồi: {cv_id} -> {jd_id} (Điểm: {score:.3f})...")

        compact_payload = CompactPayloadBuilder.build(xai)

        feedback_narrative = None
        method = "offline_fallback"

        if key_cycler:
            try:
                raw_llm = call_groq_explanation(compact_payload, key_cycler)
                feedback_narrative = CandidateFeedbackNarrative.model_validate(raw_llm).model_dump()
                method = "groq_llm"
                time.sleep(1.5)
            except Exception as exc:
                print(f"   └─ ⚠️ Lỗi Groq API ({type(exc).__name__}). Chuyển sang Universal Offline Engine...")

        if not feedback_narrative:
            feedback_narrative = generate_universal_offline_explanation(compact_payload)
            method = "offline_deterministic"

        final_schema_doc = CandidateFeedbackSchema(
            cv_id=compact_payload["cv_id"],
            jd_id=compact_payload["jd_id"],
            job_title=compact_payload["job_title"],
            mdms_score=round(compact_payload["score"], 4),
            mdms_percentage=compact_payload["score_pct"],
            feedback=CandidateFeedbackNarrative.model_validate(feedback_narrative)
        ).model_dump()

        final_results.append(final_schema_doc)

        md_text = format_candidate_feedback_report(xai, feedback_narrative)
        markdown_reports.append(md_text)

        print(f"   └─ Thành công ({method}) | Fit: {feedback_narrative['fit_level']} | Strengths: {len(feedback_narrative['strengths'])} | Gaps: {len(feedback_narrative['gaps'])}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results if len(final_results) > 1 else (final_results[0] if final_results else {}), f, indent=2, ensure_ascii=False)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(markdown_reports))

    print("\n" + "=" * 80)
    print("HOÀN THÀNH PIPELINE TẠO BÁO CÁO PHẢN HỒI HỒ SƠ")
    print(f"Tổng số báo cáo đã tạo : {len(final_results)}")
    print(f"File JSON Kết quả      : {output_json}")
    print(f"File Markdown Báo cáo  : {output_md}")
    print("=" * 80)

if __name__ == "__main__":
    main()
