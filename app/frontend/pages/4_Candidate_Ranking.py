import os
import sys

# Thêm đường dẫn thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd
import streamlit as st

# Lệnh set_page_config BẮT BUỘC phải được gọi đầu tiên
st.set_page_config(layout="wide", page_title="Ranking & Analytics", page_icon="🏆")

from app.frontend.components.charts import (
    render_ablation_chart,
    render_industry_weight_comparison,
    render_radar_chart,
)
from app.frontend.components.navbar import render_navbar
from app.frontend.utils.api_client import call_matching_api
from app.frontend.utils.mock_scoring import generate_mock_scores

# Hiển thị Navbar
render_navbar("Ranking", hr_only=True)


# =============================================================================
# POP-UP DIALOG REPORT (HÀM TẠO REPORT NỔI)
# =============================================================================
@st.dialog("📄 Candidate Assessment Report", width="large")
def show_candidate_report_dialog(candidate_name: str, scores: dict):
    """Cửa sổ Pop-up hiển thị báo cáo chi tiết giống hình 2."""
    overall_score = round(sum(scores.values()) / len(scores), 2)
    
    # Xác định mức độ phù hợp dựa trên điểm
    if overall_score >= 0.8:
        overall_fit = "Strong candidate"
    elif overall_score >= 0.65:
        overall_fit = "Moderate candidate"
    else:
        overall_fit = "Potential gap candidate"

    # Mẫu nội dung báo cáo đánh giá (Có thể thay bằng dữ liệu dynamic từ API/LLM)
    report_text = f"""Candidate: {candidate_name}
Overall fit: {overall_fit}

Strengths:
- Strong Python/ML project evidence.
- Relevant RAG and LLM implementation experience.
- Git and core ML requirements are covered.

Gaps:
- Limited evidence of Docker/cloud deployment.
- Education field is related but not a perfect match.

Relevant evidence:
- "Built RAG chatbot using LLaMA and FAISS..."
- "Fine-tuned DeepSeek-VL with LoRA..."

Interview focus:
- Ask candidate to explain evaluation methodology used for the RAG system.
- Verify deployment experience and production monitoring."""

    st.markdown("Báo cáo được trích xuất tự động từ CV và JD:")
    
    # st.code tạo khung hiển thị chuẩn dạng Code block giống hệt hình 2 (có nút Copy ở góc)
    st.code(report_text, language="text")


# -----------------------------------------------------------------------------
# TIÊU ĐỀ TRANG
# -----------------------------------------------------------------------------
st.title("🏆 Candidate Ranking & 📊 Analytics Dashboard")
st.markdown("Xếp hạng ứng viên dựa trên các CV đã upload & xác nhận ở bước **Upload CV**.")

st.divider()

# =============================================================================
# PHẦN 1: CANDIDATE RANKING (XẾP HẠNG ỨNG VIÊN)
# =============================================================================
st.header("🏆 1. Xếp hạng ứng viên (Candidate Ranking)")

if not st.session_state.get("jd_confirmed_features"):
    st.warning("⚠️ Vui lòng hoàn tất Upload → Extract → Refine → Confirm ở Job Description trước.")
    st.stop()

candidates = st.session_state.get("cv_confirmed_candidates")
if not candidates:
    st.warning("⚠️ Vui lòng hoàn tất Upload → Extract → Refine → Confirm ở CV trước (mục 'Upload CV').")
    st.stop()

st.info(f"Đang có **{len(candidates)}** ứng viên đã xác nhận sẵn sàng để xếp hạng.")

if st.button("🔍 Chạy xếp hạng", type="primary"):
    results = []
    for cand in candidates:
        scores = generate_mock_scores(cand["filename"])
        overall = round(sum(scores.values()) / len(scores), 3)
        results.append({
            "Tên ứng viên (file)": cand["filename"],
            "Điểm tổng thể": overall,
            **scores
        })

    df = pd.DataFrame(results).sort_values(by="Điểm tổng thể", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Hạng"

    st.session_state["ranking_df"] = df

if "ranking_df" in st.session_state:
    df = st.session_state["ranking_df"]

    st.subheader("📋 Bảng xếp hạng ứng viên")

    df_display = df.copy()
    df_display.insert(0, "Chọn", False)

    edited_df = st.data_editor(
        df_display,
        column_config={
            "Chọn": st.column_config.CheckboxColumn(required=True),
            "Điểm tổng thể": st.column_config.ProgressColumn(
                "Điểm tổng thể", min_value=0, max_value=1, format="%.2f"
            ),
        },
        disabled=[c for c in df_display.columns if c != "Chọn"],
        use_container_width=True,
    )

    st.divider()

    # -------------------------------------------------------------------------
    # XEM CHI TIẾT & GENERATE REPORT
    # -------------------------------------------------------------------------
    st.subheader("🔍 Xem chi tiết & Tạo báo cáo ứng viên")
    
    col_select, col_btn = st.columns([3, 1], vertical_alignment="bottom")
    
    with col_select:
        selected_candidate = st.selectbox("Chọn ứng viên", df["Tên ứng viên (file)"])
        
    candidate_row = df[df["Tên ứng viên (file)"] == selected_candidate].iloc[0]
    detail_scores = {
        "Edu Score": candidate_row["Edu Score"],
        "Skill Score": candidate_row["Skill Score"],
        "Domain Score": candidate_row["Domain Score"],
        "Exp Score": candidate_row["Exp Score"],
    }

    with col_btn:
        # Nút nhấn để kích hoạt Pop-up
        if st.button("📝 Generate Report", type="secondary", use_container_width=True):
            show_candidate_report_dialog(selected_candidate, detail_scores)

    # Hiển thị biểu đồ Radar bên dưới
    render_radar_chart(detail_scores, title=f"Radar Chart — {selected_candidate}")

    st.divider()

    # Shortlist management
    shortlisted = edited_df[edited_df["Chọn"] == True]

    if not shortlisted.empty:
        st.subheader(f"✅ Đã chọn {len(shortlisted)} ứng viên vào shortlist")
        st.dataframe(shortlisted.drop(columns=["Chọn"]), use_container_width=True)

        csv = shortlisted.drop(columns=["Chọn"]).to_csv(index=True).encode("utf-8-sig")
        st.download_button(
            "📥 Tải shortlist (CSV)",
            data=csv,
            file_name="shortlist_ung_vien.csv",
            mime="text/csv"
        )
    else:
        st.caption("Tick chọn ứng viên ở bảng trên để đưa vào shortlist.")


# =============================================================================
# PHẦN 2: ANALYTICS DASHBOARD
# =============================================================================
st.divider()
st.header("📊 2. Analytics Dashboard")

st.subheader("2.1. Ablation Impact Chart")
mock_ablation_data = {
    "Full Model": 0.85,
    "Bỏ Edu Score": 0.79,
    "Bỏ Skill Score": 0.68,
    "Bỏ Domain Score": 0.75,
    "Bỏ Exp Score": 0.71,
}
render_ablation_chart(mock_ablation_data)

st.divider()

st.subheader("2.2. So sánh trọng số theo ngành")
mock_industry_data = {
    "IT": {"Edu Score": 0.15, "Skill Score": 0.50, "Domain Score": 0.20, "Exp Score": 0.15},
    "Sales": {"Edu Score": 0.10, "Skill Score": 0.20, "Domain Score": 0.20, "Exp Score": 0.50},
    "Finance": {"Edu Score": 0.30, "Skill Score": 0.25, "Domain Score": 0.25, "Exp Score": 0.20},
}
render_industry_weight_comparison(mock_industry_data)