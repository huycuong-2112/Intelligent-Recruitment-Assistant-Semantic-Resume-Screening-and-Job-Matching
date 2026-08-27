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

HARDCODED_GROQ_API_KEY = ""

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
    """Tự động lấy danh sách model chat khả dụng cho API Key."""
    try:
        models_data = client.models.list().data
        # Lọc bỏ whisper (audio) và guard (moderation), chỉ lấy model sinh văn bản
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
# 2. PATH CONFIGURATION
# ---------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

POTENTIAL_INPUT_PATHS = [
    project_root / "Data" / "Results" / "IT" / "XAI" / "Examples" / "jd_001_cv_example_xai.json",
]

DEFAULT_OUTPUT_JSON = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001" / "final_explanation.json"
DEFAULT_OUTPUT_MD = project_root / "Data" / "Results" / "IT" / "EvaluationReports" / "jd_001" / "candidate_explanations.md"

# ---------------------------------------------------------------------------
# 3. SCHEMA DEFINITIONS (FINAL EXPLANATION V1)
# ---------------------------------------------------------------------------
class StrengthItem(BaseModel):
    text: str = Field(..., description="Mô tả điểm mạnh")
    evidence_refs: List[str] = Field(default_factory=list, description="Danh sách evidence_id tương ứng")

class GapItem(BaseModel):
    type: Literal["required_skill_no_evidence", "weak_experience_evidence", "other"] = Field(
        ..., description="Phân loại khoảng trống bằng chứng"
    )
    text: str = Field(..., description="Mô tả khách quan về khoảng trống bằng chứng")
    evidence_refs: List[str] = Field(default_factory=list, description="Danh sách evidence_id nếu có")

class InterviewFocusItem(BaseModel):
    topic: str = Field(..., description="Chủ đề / Kỹ năng cần phỏng vấn")
    question: str = Field(..., description="Câu hỏi phỏng vấn đề xuất")
    reason: str = Field(..., description="Lý do cần hỏi dựa trên gap hoặc bằng chứng yếu")
    evidence_refs: List[str] = Field(default_factory=list, description="evidence_id liên quan")

class LLMNarrativeOutput(BaseModel):
    summary: str = Field(..., description="Tóm tắt 2-3 câu về mức độ phù hợp và lý do chính")
    strengths: List[StrengthItem] = Field(default_factory=list)
    gaps: List[GapItem] = Field(default_factory=list)
    interview_focus: List[InterviewFocusItem] = Field(default_factory=list)

class DimensionsScore(BaseModel):
    skill: float
    experience: float
    education: float
    semantic: float

class FinalDecision(BaseModel):
    final_score: float
    coverage: float
    dimensions: DimensionsScore

class ExplanationBlock(BaseModel):
    summary: str
    strengths: List[StrengthItem]
    gaps: List[GapItem]
    interview_focus: List[InterviewFocusItem]
    disclaimer: str = (
        "Không có bằng chứng trong CV không đồng nghĩa ứng viên chắc chắn không có năng lực; "
        "điểm số phản ánh bằng chứng trích xuất được so với yêu cầu JD."
    )

class FinalExplanationSchema(BaseModel):
    schema_version: str = "explanation_v1"
    source_xai_schema_version: str = "xai_v1"
    cv_id: str
    jd_id: str
    target_role: str
    model_version: str = "mdms_tuned_v1"
    decision: FinalDecision
    explanation: ExplanationBlock

# ---------------------------------------------------------------------------
# 4. C3: COMPACT PAYLOAD BUILDER (DETERMINISTIC PYTHON FILTER)
# ---------------------------------------------------------------------------
class CompactPayloadBuilder:
    @staticmethod
    def build(xai_data: Dict[str, Any]) -> Dict[str, Any]:
        cv_id = xai_data.get("cv_id") or xai_data.get("metadata", {}).get("cv_id", "N/A")
        jd_id = xai_data.get("jd_id") or xai_data.get("metadata", {}).get("jd_id", "N/A")
        job_title = xai_data.get("job_title") or xai_data.get("metadata", {}).get("job_title", "N/A")

        registry = xai_data.get("evidence_registry", {})
        dims = xai_data.get("dimensions", {})

        # 1. Strengths kèm evidence_refs
        strengths = []
        for s in xai_data.get("strength_candidates", [])[:3]:
            stype = s.get("type", "")
            eref = s.get("evidence_ref")
            ev_refs = [eref] if eref and eref in registry else []
            ev_text = registry.get(eref, {}).get("source_text", "").strip() if eref else ""

            if stype == "degree_match":
                deg_cand = dims.get("education", {}).get("degree", {}).get("candidate", "Degree")
                field_cand = dims.get("education", {}).get("field", {}).get("candidate", "")
                strengths.append({
                    "fact": f"Bằng cấp phù hợp yêu cầu ({deg_cand} — {field_cand})",
                    "evidence_refs": ev_refs,
                    "evidence_snippet": ev_text
                })
            else:
                strengths.append({
                    "fact": s.get("fact") or stype.replace("_", " ").title(),
                    "evidence_refs": ev_refs,
                    "evidence_snippet": ev_text
                })

        exp_years = dims.get("experience", {}).get("years", {}).get("candidate_years")
        min_years = dims.get("experience", {}).get("years", {}).get("minimum_years", 0.0)
        if exp_years is not None and exp_years >= min_years and len(strengths) < 3:
            work_refs = [eid for eid, edata in registry.items() if edata.get("source_type") == "cv_work_experience"]
            strengths.append({
                "fact": f"Đáp ứng số năm kinh nghiệm ({exp_years} năm so với yêu cầu tối thiểu {min_years} năm)",
                "evidence_refs": work_refs[:1],
                "evidence_snippet": f"Đã xác minh {exp_years} năm kinh nghiệm làm việc"
            })

        # 2. Đếm số lượng Missing Skills thực tế
        missing_skills = dims.get("skill", {}).get("missing_required", [])
        missing_skill_count = len(missing_skills)

        # 3. Trích xuất Weak Experience
        weak_exp = []
        for g in xai_data.get("gap_candidates", []):
            if g.get("dimension") == "experience" and len(weak_exp) < 2:
                eref = g.get("evidence_ref")
                ev_refs = [eref] if eref and eref in registry else []
                ev_text = registry.get(eref, {}).get("source_text", "").strip() if eref else ""
                weak_exp.append({
                    "requirement": g.get("requirement", "Trách nhiệm chuyên môn"),
                    "evidence_refs": ev_refs,
                    "evidence_snippet": ev_text
                })

        weak_exp_count = len(weak_exp)

        # 4. Tính toán quy tắc số lượng câu hỏi phỏng vấn
        total_gaps = missing_skill_count + weak_exp_count
        if total_gaps == 0:
            max_interview_questions = 1
        elif total_gaps <= 2:
            max_interview_questions = 2
        else:
            max_interview_questions = 3

        # 5. Xây dựng danh sách chủ đề phỏng vấn đủ số lượng max_questions
        suggested_topics = []
        seen_topics = set()

        # Lấy từ interview_focus gốc trước
        for f in xai_data.get("interview_focus", []):
            tname = f.get("topic")
            eref = f.get("evidence_ref")
            if tname and tname not in seen_topics:
                seen_topics.add(tname)
                suggested_topics.append({
                    "topic": tname,
                    "reason_hint": f.get("reason", "Làm rõ mức độ thành thạo thực tế"),
                    "evidence_refs": [eref] if eref and eref in registry else []
                })

        # Bổ sung từ missing skills nếu chưa đủ số lượng
        for s in missing_skills:
            if len(suggested_topics) >= max_interview_questions:
                break
            if s not in seen_topics:
                seen_topics.add(s)
                suggested_topics.append({
                    "topic": s,
                    "reason_hint": f"Chưa tìm thấy bằng chứng cho kỹ năng bắt buộc: {s}",
                    "evidence_refs": []
                })

        # Bổ sung từ weak experience nếu vẫn chưa đủ
        for w in weak_exp:
            if len(suggested_topics) >= max_interview_questions:
                break
            req = w.get("requirement")
            if req not in seen_topics:
                seen_topics.add(req)
                suggested_topics.append({
                    "topic": req,
                    "reason_hint": "Bằng chứng hiện tại còn hạn chế so với yêu cầu trách nhiệm",
                    "evidence_refs": w.get("evidence_refs", [])
                })

        return {
            "candidate": {
                "cv_id": cv_id,
                "jd_id": jd_id,
                "job_title": job_title
            },
            "strengths": strengths,
            "gaps": {
                "missing_required_skills": missing_skills,
                "missing_skill_count": missing_skill_count,
                "weak_experience": weak_exp,
                "weak_experience_count": weak_exp_count,
                "total_gaps": total_gaps
            },
            "interview_config": {
                "max_questions": max_interview_questions,
                "suggested_topics": suggested_topics[:max_interview_questions]
            }
        }

# ---------------------------------------------------------------------------
# 5. C4: GROQ PRODUCTION PROMPT & LLM CALL
# ---------------------------------------------------------------------------
GROQ_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên tạo nội dung giải thích kết quả matching CV–JD cho chuyên viên tuyển dụng (HR).
Dựa DUY NHẤT vào JSON facts được cung cấp, hãy tạo báo cáo giải thích bằng tiếng Việt.

QUY TẮC WORDING & EVIDENCE TRACEABILITY (BẮT BUỘC):
1. Tính khách quan XAI: Tuyệt đối KHÔNG phán xét chủ quan năng lực ứng viên (VD: KHÔNG dùng "ứng viên ít kinh nghiệm", "ứng viên yếu", "thiếu năng lực").
2. Bám sát bằng chứng: Diễn đạt chuẩn mực (VD: "Bằng chứng hiện tại cho thấy mức độ phù hợp còn hạn chế với một số trách nhiệm...", "Chưa tìm thấy bằng chứng đề cập đến kỹ năng X trong hồ sơ").
3. Map đúng evidence_refs:
   - Trong `strengths`: Phải giữ nguyên mảng `evidence_refs` từ input.
   - Trong `gaps`:
     * Với kỹ năng thiếu: `type` = "required_skill_no_evidence", `evidence_refs`: []
     * Với kinh nghiệm yếu: `type` = "weak_experience_evidence", `evidence_refs`: [evidence_id tương ứng]
   - Trong `interview_focus`: `evidence_refs` lấy từ gợi ý hoặc để [] nếu là skill mới.
4. Giới hạn số câu hỏi: Danh sách `interview_focus` tối đa đúng bằng `max_questions` trong config.
5. Giữ nguyên tên công nghệ và thuật ngữ tiếng Anh (Python, Git, Docker, Machine Learning, etc.).

CẤU TRÚC JSON ĐẦU RA:
{
  "summary": "Tóm tắt 2-3 câu về độ phù hợp tổng quan theo góc nhìn bằng chứng.",
  "strengths": [
    {
      "text": "Mô tả điểm mạnh cụ thể",
      "evidence_refs": ["ev_..."]
    }
  ],
  "gaps": [
    {
      "type": "required_skill_no_evidence",
      "text": "Chưa tìm thấy bằng chứng cho các kỹ năng bắt buộc: ...",
      "evidence_refs": []
    },
    {
      "type": "weak_experience_evidence",
      "text": "Bằng chứng hiện tại cho thấy mức độ phù hợp còn hạn chế với trách nhiệm '...'",
      "evidence_refs": ["ev_..."]
    }
  ],
  "interview_focus": [
    {
      "topic": "Tên chủ đề / kỹ năng",
      "question": "Câu hỏi phỏng vấn đề xuất",
      "reason": "Lý do cần hỏi dựa trên khoảng trống dữ liệu",
      "evidence_refs": []
    }
  ]
}
"""

def call_groq_explanation(compact_payload: Dict[str, Any], key_cycler: Any) -> Dict[str, Any]:
    user_prompt = (
        "Respond with a valid JSON object only.\n"
        "INPUT COMPACT PAYLOAD:\n" + json.dumps(compact_payload, ensure_ascii=False, indent=2)
    )
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
                wait_match = re.search(r"in\s+([0-9\.]+)\s*s", str(rle))
                wait_time = float(wait_match.group(1)) + 0.5 if wait_match else (3.0 * (2 ** (attempt - 1)))
                print(f"   └─ ⏳ Rate limit 429 trên model {model_name}. Chờ {wait_time:.1f}s...")
                time.sleep(wait_time)
                break
            except Exception as exc:
                last_error = exc
                # In chi tiết lỗi thật từ server Groq để debug ngay lập tức
                error_msg = getattr(exc, "message", str(exc))
                print(f"   └─ ⚠️ Model {model_name} bị lỗi: {error_msg}")
                continue

    raise last_error or RuntimeError("Groq explanation call failed across all models.")

# ---------------------------------------------------------------------------
# 6. DETERMINISTIC OFFLINE GENERATOR (FALLBACK KHI KHÔNG CÓ API KEY)
# ---------------------------------------------------------------------------
def generate_offline_explanation(compact_payload: Dict[str, Any]) -> Dict[str, Any]:
    cand = compact_payload.get("candidate", {})
    strengths_raw = compact_payload.get("strengths", [])
    gaps_raw = compact_payload.get("gaps", {})
    missing_skills = gaps_raw.get("missing_required_skills", [])
    weak_exp = gaps_raw.get("weak_experience", [])
    interview_cfg = compact_payload.get("interview_config", {})
    max_q = interview_cfg.get("max_questions", 2)

    summary = (
        f"Hồ sơ ứng viên {cand.get('cv_id')} đáp ứng tiêu chuẩn về bằng cấp và thời gian kinh nghiệm đối với vị trí {cand.get('job_title')}. "
        f"Tuy nhiên, quá trình trích xuất hồ sơ chưa ghi nhận bằng chứng cho {len(missing_skills)} kỹ năng bắt buộc, "
        f"và bằng chứng hiện tại cho thấy mức độ phù hợp còn hạn chế với một số trách nhiệm triển khai AI/ML chuyên sâu."
    )

    strengths = [{"text": s.get("fact"), "evidence_refs": s.get("evidence_refs", [])} for s in strengths_raw]

    gaps = []
    if missing_skills:
        gaps.append({
            "type": "required_skill_no_evidence",
            "text": f"Chưa tìm thấy bằng chứng đề cập đến các kỹ năng bắt buộc: {', '.join(missing_skills[:4])}.",
            "evidence_refs": []
        })
    for w in weak_exp:
        gaps.append({
            "type": "weak_experience_evidence",
            "text": f"Bằng chứng hiện tại cho thấy mức độ phù hợp còn hạn chế với trách nhiệm '{w.get('requirement')}'.",
            "evidence_refs": w.get("evidence_refs", [])
        })

    # Đa dạng hóa câu hỏi theo từng nhóm chủ đề
    question_templates = {
        "Python programming": "Khi làm việc với các tập dữ liệu lớn trong Python, bạn thường dùng những kỹ thuật hoặc thư viện nào để tối ưu bộ nhớ và tăng tốc độ xử lý?",
        "Data structures and algorithms": "Bạn có thể lấy ví dụ về một bài toán thực tế mà bạn đã chủ động lựa chọn cấu trúc dữ liệu phù hợp để giảm độ phức tạp thuật toán không?",
        "Version control (Git)": "Bạn quản lý quy trình phân nhánh (branching strategy) và xử lý xung đột mã nguồn (merge conflicts) phức tạp trong nhóm như thế nào?",
        "Basic Machine Learning / Deep Learning concepts": "Khi mô hình huấn luyện gặp hiện tượng Overfitting hoặc mất cân bằng dữ liệu nghiêm trọng, bạn áp dụng những phương pháp nào để khắc phục?",
    }

    interview_focus = []
    for topic_item in interview_cfg.get("suggested_topics", [])[:max_q]:
        tname = topic_item.get("topic")
        # Chọn câu hỏi theo template ngữ cảnh hoặc câu hỏi trách nhiệm
        if tname in question_templates:
            q_text = question_templates[tname]
            r_text = f"Kiểm tra chiều sâu tư duy kỹ thuật và khả năng tối ưu thực chiến đối với {tname}."
        elif "Research and implement" in tname or "AI projects" in tname:
            q_text = "Trong các dự án trước đây, bạn trực tiếp tham gia xây dựng/tinh chỉnh thuật toán AI ở mức độ nào, hay chủ yếu điều phối vòng đời dự án?"
            r_text = f"Làm rõ ranh giới đóng góp kỹ thuật trực tiếp do bằng chứng {topic_item.get('evidence_refs', [''])[0]} thiên về quản lý dự án."
        else:
            q_text = f"Trong một bài toán thực tế yêu cầu áp dụng {tname}, bạn sẽ tiếp cận các bước thiết kế và giải quyết vấn đề như thế nào?"
            r_text = f"Xác minh khả năng áp dụng thực tế đối với {tname}."

        interview_focus.append({
            "topic": tname,
            "question": q_text,
            "reason": r_text,
            "evidence_refs": topic_item.get("evidence_refs", [])
        })

    return {
        "summary": summary,
        "strengths": strengths,
        "gaps": gaps,
        "interview_focus": interview_focus
    }
# ---------------------------------------------------------------------------
# 7. CODE CLEAN & ASSEMBLE (STRICT SCHEMA VALIDATION & HARD SLICING)
# ---------------------------------------------------------------------------
def assemble_final_clean_schema(
    xai_data: Dict[str, Any],
    llm_raw_data: Dict[str, Any],
    max_questions_allowed: int
) -> Dict[str, Any]:
    decision = xai_data.get("decision", {})
    dims = xai_data.get("dimensions", {})
    cv_id = xai_data.get("cv_id") or xai_data.get("metadata", {}).get("cv_id", "N/A")
    jd_id = xai_data.get("jd_id") or xai_data.get("metadata", {}).get("jd_id", "N/A")
    job_title = xai_data.get("job_title") or xai_data.get("metadata", {}).get("job_title", "N/A")

    # Hard-guardrail: Cắt tỉa số lượng interview questions bằng code nếu LLM sinh dư
    interview_items = llm_raw_data.get("interview_focus", [])
    if len(interview_items) > max_questions_allowed:
        interview_items = interview_items[:max_questions_allowed]
        llm_raw_data["interview_focus"] = interview_items

    # Validate LLM narrative portion
    validated_narrative = LLMNarrativeOutput.model_validate(llm_raw_data)

    final_obj = FinalExplanationSchema(
        schema_version="explanation_v1",
        source_xai_schema_version="xai_v1",
        cv_id=cv_id,
        jd_id=jd_id,
        target_role=job_title,
        model_version="mdms_tuned_v1",
        decision=FinalDecision(
            final_score=round(decision.get("final_score", 0.0), 4),
            coverage=round(decision.get("coverage", 1.0), 2),
            dimensions=DimensionsScore(
                skill=round(dims.get("skill", {}).get("score", 0.0), 4),
                experience=round(dims.get("experience", {}).get("score", 0.0), 4),
                education=round(dims.get("education", {}).get("score", 0.0), 4),
                semantic=round(dims.get("semantic", {}).get("score", 0.0), 4),
            )
        ),
        explanation=ExplanationBlock(
            summary=validated_narrative.summary,
            strengths=validated_narrative.strengths,
            gaps=validated_narrative.gaps,
            interview_focus=validated_narrative.interview_focus,
            disclaimer="Không có bằng chứng trong CV không đồng nghĩa ứng viên chắc chắn không có năng lực; điểm số phản ánh bằng chứng trích xuất được so với yêu cầu JD."
        )
    )

    return final_obj.model_dump()

def format_markdown_report(final_data: Dict[str, Any]) -> str:
    dec = final_data.get("decision", {})
    dims = dec.get("dimensions", {})
    exp = final_data.get("explanation", {})

    # 1. Render Strengths
    strengths_lines = []
    for s in exp.get("strengths", []):
        refs_str = f" *(Refs: {', '.join(s.get('evidence_refs', []))})*" if s.get("evidence_refs") else ""
        strengths_lines.append(f"- {s.get('text')}{refs_str}")
    strengths_md = "\n".join(strengths_lines) if strengths_lines else "- Không ghi nhận điểm mạnh nổi bật."

    # 2. Render Gaps (Khắc phục lỗi mất trắng)
    gaps_lines = []
    for g in exp.get("gaps", []):
        refs_str = f" *(Refs: {', '.join(g.get('evidence_refs', []))})*" if g.get("evidence_refs") else ""
        gtype = g.get("type", "gap").upper()
        gaps_lines.append(f"- [{gtype}] {g.get('text')}{refs_str}")
    gaps_md = "\n".join(gaps_lines) if gaps_lines else "- Không phát hiện khoảng trống đáng kể."

    # 3. Render Interview Focus
    interview_lines = []
    for q in exp.get("interview_focus", []):
        refs_str = f" *(Refs: {', '.join(q.get('evidence_refs', []))})*" if q.get("evidence_refs") else ""
        interview_lines.append(
            f"- **Chủ đề:** `{q.get('topic')}`\n"
            f"  **Câu hỏi:** *{q.get('question')}*\n"
            f"  **Lý do:** {q.get('reason')}{refs_str}"
        )
    interview_md = "\n".join(interview_lines) if interview_lines else "- Không có câu hỏi phỏng vấn đề xuất."

    md = (
        f"# Báo cáo Đánh giá & Giải thích Kết quả Tuyển dụng\n\n"
        f"**Mã ứng viên:** `{final_data.get('cv_id')}`  \n"
        f"**Vị trí ứng tuyển:** `{final_data.get('target_role')}` (`{final_data.get('jd_id')}`)  \n"
        f"**Phiên bản model:** `{final_data.get('model_version')}`  \n"
        f"**Điểm MDMS tổng hợp:** **`{dec.get('final_score', 0.0):.4f}`** (Độ phủ: `{dec.get('coverage', 1.0) * 100:.0f}%`)\n\n"
        f"### Điểm Thành phần:\n"
        f"* **Kỹ năng (Skill - 40%):** `{dims.get('skill', 0.0):.3f}`\n"
        f"* **Kinh nghiệm (Experience - 20%):** `{dims.get('experience', 0.0):.3f}`\n"
        f"* **Học vấn (Education - 10%):** `{dims.get('education', 0.0):.3f}`\n"
        f"* **Ngữ nghĩa tổng thể (Semantic - 30%):** `{dims.get('semantic', 0.0):.3f}`\n\n"
        f"---\n\n"
        f"## 1. Tóm tắt Đánh giá\n"
        f"{exp.get('summary')}\n\n"
        f"## 2. Điểm mạnh Nổi bật\n"
        f"{strengths_md}\n\n"
        f"## 3. Khoảng trống & Bằng chứng còn yếu\n"
        f"{gaps_md}\n\n"
        f"## 4. Trọng tâm Phỏng vấn Đề xuất\n"
        f"{interview_md}\n\n"
        f"---\n\n"
        f"> *Lưu ý: {exp.get('disclaimer')}*"
    )
    return md

# ---------------------------------------------------------------------------
# 8. RESOLVE INPUT FILE / DIRECTORY
# ---------------------------------------------------------------------------
def find_input_source() -> Path:
    for candidate in POTENTIAL_INPUT_PATHS:
        if candidate.exists():
            return candidate
    sample_path = current_file.parent / "jd_001_cv_example_xai.json"
    if sample_path.exists():
        return sample_path
    raise FileNotFoundError("Không tìm thấy file XAI input JSON.")

# ---------------------------------------------------------------------------
# 9. MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PIPELINE TẠO BÁO CÁO GIẢI THÍCH MATCHING (C3 COMPACT -> C4 GROQ -> FINAL)")
    print(f"Project Root   : {project_root}")

    input_path = find_input_source()
    output_json = DEFAULT_OUTPUT_JSON
    output_md = DEFAULT_OUTPUT_MD
    output_json.parent.mkdir(parents=True, exist_ok=True)

    key_cycler = get_groq_client_cycler()
    mode = "ONLINE (Groq Cloud LLM)" if key_cycler else "OFFLINE (Deterministic Fallback Engine)"

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

        print(f"[{idx}/{len(items)}] Đang xử lý giải thích: {cv_id} -> {jd_id} (Điểm: {score:.3f})...")

        # C3: Nén facts & tính count trong Python
        compact_payload = CompactPayloadBuilder.build(xai)
        max_q = compact_payload.get("interview_config", {}).get("max_questions", 2)

        # C4: Gọi Groq LLM (hoặc offline fallback)
        llm_narrative = None
        method = "offline_fallback"

        if key_cycler:
            try:
                llm_narrative = call_groq_explanation(compact_payload, key_cycler)
                method = "groq_llm"
                time.sleep(1.5)
            except Exception as exc:
                print(f"   └─ ⚠️ Lỗi Groq API ({type(exc).__name__}). Chuyển sang Offline Engine...")

        if not llm_narrative:
            llm_narrative = generate_offline_explanation(compact_payload)
            method = "offline_deterministic"

        # Clean & Validate bằng Code
        final_doc = assemble_final_clean_schema(xai, llm_narrative, max_questions_allowed=max_q)
        final_results.append(final_doc)

        md_text = format_markdown_report(final_doc)
        markdown_reports.append(md_text)

        print(f"   └─ Thành công ({method}) | Strengths: {len(final_doc['explanation']['strengths'])} | Gaps: {len(final_doc['explanation']['gaps'])} | Questions: {len(final_doc['explanation']['interview_focus'])}")

    # Ghi file JSON & MD
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results if len(final_results) > 1 else final_results[0], f, indent=2, ensure_ascii=False)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(markdown_reports))

    print("\n" + "=" * 80)
    print("HOÀN THÀNH PIPELINE TẠO BÁO CÁO GIẢI THÍCH")
    print(f"Tổng số báo cáo đã tạo : {len(final_results)}")
    print(f"File JSON Kết quả      : {output_json}")
    print(f"File Markdown Báo cáo  : {output_md}")
    print("=" * 80)

if __name__ == "__main__":
    main()