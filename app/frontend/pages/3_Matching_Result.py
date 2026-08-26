import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
from app.frontend.components.navbar import render_navbar
from app.frontend.utils.api_client import call_matching_api
from app.frontend.utils.mock_extraction import group_by_category

render_navbar("Rating")

st.title("🔍 Kết quả Matching")

cv_candidates = st.session_state.get("cv_confirmed_candidates")
jd_features = st.session_state.get("jd_confirmed_features")

if not cv_candidates or not jd_features:
    st.warning("Vui lòng hoàn tất Upload → Extract → Refine → Confirm ở cả CV và Job Description trước.")
    st.stop()

# =========================
# JD
# =========================

job_description = ", ".join(
    f["name"] for f in jd_features
)

with st.expander("📋 Feature JD đã xác nhận"):
    st.write(group_by_category(jd_features))

# =========================
# CV list
# =========================

st.subheader(f"👥 {len(cv_candidates)} CV sẽ được đánh giá")

for candidate in cv_candidates:
    with st.expander(f"📄 {candidate['filename']}", expanded=False):
        st.write(group_by_category(candidate["confirmed_features"]))

# =========================
# Matching
# =========================

if st.button("🚀 Tính độ tương đồng cho tất cả CV", type="primary", use_container_width=True):
    results = []

    progress = st.progress(0)

    for i, candidate in enumerate(cv_candidates):
        resume_text = ", ".join(
            f["name"] for f in candidate["confirmed_features"]
        )

        with st.spinner(f"Đang xử lý {candidate['filename']}..."):
            try:
                result = call_matching_api(
                    resume_text,
                    job_description
                )

                results.append({
                    "filename": candidate["filename"],
                    "similarity_score": result["similarity_score"],
                    "explanation": result.get("explanation", "")
                })

            except Exception as e:
                results.append({
                    "filename": candidate["filename"],
                    "similarity_score": None,
                    "explanation": f"Lỗi: {e}"
                })

        progress.progress((i + 1) / len(cv_candidates))

    results.sort(
        key=lambda x: x["similarity_score"] if x["similarity_score"] is not None else -1,
        reverse=True
    )

    st.session_state["rating_results"] = results

# =========================
# Results
# =========================

results = st.session_state.get("rating_results")

if results:
    st.divider()
    st.subheader("🏆 Ranking kết quả")

    for rank, result in enumerate(results, start=1):
        score = result["similarity_score"]

        if score is None:
            score_text = "Lỗi"
        else:
            score_text = f"{score * 100:.1f}%"

        col_rank, col_name, col_score = st.columns([1, 6, 2])

        with col_rank:
            st.markdown(f"### #{rank}")

        with col_name:
            st.markdown(f"**📄 {result['filename']}**")

        with col_score:
            st.metric("Similarity", score_text)

        if result["explanation"]:
            st.caption(result["explanation"])

    # =========================
    # Thống kê & Tổng hợp (Mới bổ sung)
    # =========================
    st.divider()
    st.subheader("📊 Thống kê & Tổng hợp chi tiết")

    valid_results = [r for r in results if r["similarity_score"] is not None]

    if valid_results:
        total_cvs = len(results)
        valid_cvs = len(valid_results)
        
        # Tính toán các chỉ số thống kê
        avg_score = sum(r["similarity_score"] for r in valid_results) / valid_cvs
        highest_cv = max(valid_results, key=lambda x: x["similarity_score"])
        lowest_cv = min(valid_results, key=lambda x: x["similarity_score"])
        
        # Phân loại số lượng theo ngưỡng
        high_match_count = sum(1 for r in valid_results if r["similarity_score"] >= 0.7)
        medium_match_count = sum(1 for r in valid_results if 0.5 <= r["similarity_score"] < 0.7)
        low_match_count = sum(1 for r in valid_results if r["similarity_score"] < 0.5)

        # Hiển thị các thẻ chỉ số KPI
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        with col_kpi1:
            st.metric(
                label="Tổng số CV đã đánh giá", 
                value=f"{total_cvs} CV",
                delta=f"{valid_cvs} thành công" if valid_cvs == total_cvs else f"{total_cvs - valid_cvs} lỗi",
                delta_color="normal" if valid_cvs == total_cvs else "inverse"
            )

        with col_kpi2:
            st.metric(
                label="Điểm trung bình", 
                value=f"{avg_score * 100:.1f}%"
            )

        with col_kpi3:
            st.metric(
                label="Cao nhất", 
                value=f"{highest_cv['similarity_score'] * 100:.1f}%",
                help=f"CV: {highest_cv['filename']}"
            )

        with col_kpi4:
            st.metric(
                label="Ứng viên tiềm năng (≥70%)", 
                value=f"{high_match_count}/{valid_cvs}",
                delta=f"{(high_match_count / valid_cvs) * 100:.0f}% tổng số CV"
            )

        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

        # Bảng tổng hợp kèm phân loại mức độ phù hợp
        table_data = []

        for rank, result in enumerate(results, start=1):
            score = result["similarity_score"]
            
            if score is None:
                score_str = "Lỗi"
                rating_level = "❌ Lỗi"
            else:
                score_str = f"{score * 100:.1f}%"
                if score >= 0.7:
                    rating_level = "🟢 Phù hợp cao"
                elif score >= 0.5:
                    rating_level = "🟡 Trung bình"
                else:
                    rating_level = "🔴 Thấp"

            table_data.append({
                "Hạng": rank,
                "Tên CV": result["filename"],
                "Độ tương thích": score_str,
                "Đánh giá": rating_level
            })

        st.dataframe(table_data, use_container_width=True, hide_index=True)

    else:
        st.error("Không có CV nào được xử lý thành công để hiển thị thống kê.")