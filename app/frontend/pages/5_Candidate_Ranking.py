import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
import pandas as pd

from app.frontend.components.charts import render_radar_chart
from app.frontend.utils.mock_scoring import generate_mock_scores

st.set_page_config(layout="wide")
st.title("🏆 Candidate Ranking")
st.markdown("Upload nhiều CV để so khớp với Job Description và xem bảng xếp hạng ứng viên.")

st.divider()

if "job_description" not in st.session_state:
    st.warning("Vui lòng upload Job Description trước ở trang 'Upload JobDescription'.")
    st.stop()

st.info("Đang so khớp với Job Description hiện tại.")

uploaded_files = st.file_uploader(
    "Upload nhiều CV ứng viên (PDF/DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🔍 Chạy xếp hạng"):
        results = []
        for f in uploaded_files:
            scores = generate_mock_scores(f.name)
            overall = round(sum(scores.values()) / len(scores), 3)
            results.append({
                "Tên ứng viên (file)": f.name,
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

    st.subheader("🔍 Xem chi tiết ứng viên")
    selected_candidate = st.selectbox("Chọn ứng viên để xem Radar Chart", df["Tên ứng viên (file)"])

    if selected_candidate:
        candidate_row = df[df["Tên ứng viên (file)"] == selected_candidate].iloc[0]
        detail_scores = {
            "Edu Score": candidate_row["Edu Score"],
            "Skill Score": candidate_row["Skill Score"],
            "Domain Score": candidate_row["Domain Score"],
            "Exp Score": candidate_row["Exp Score"],
        }
        render_radar_chart(detail_scores, title=f"Radar Chart — {selected_candidate}")

    st.divider()

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