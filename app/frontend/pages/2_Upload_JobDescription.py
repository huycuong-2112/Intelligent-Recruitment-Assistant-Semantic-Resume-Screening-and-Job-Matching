import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st

st.title("💼 Upload Job Description")

uploaded_jd = st.file_uploader(
    "Chọn file Job Description (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"]
)

if uploaded_jd is not None:
    st.success(f"Đã upload: {uploaded_jd.name}")
    # TODO: gọi API để parse text từ file JD (giống luồng xử lý CV)
    # Hiện tại dùng placeholder để test luồng UI
    st.session_state["job_description"] = "Nội dung JD mẫu (placeholder, sẽ được trích xuất từ file thật)"

if "job_description" in st.session_state:
    st.text_area(
        "Nội dung JD đã trích xuất",
        st.session_state["job_description"],
        height=200,
        disabled=True
    )