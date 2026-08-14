import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
from app.frontend.utils.api_client import call_matching_api

st.title("📄 Upload CV")

uploaded_file = st.file_uploader("Chọn file CV (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    st.success(f"Đã upload: {uploaded_file.name}")
    # TODO: gọi API để parse text từ file CV
    st.session_state["resume_text"] = "Nội dung CV mẫu (placeholder)"

if "resume_text" in st.session_state:
    st.text_area("Nội dung CV đã trích xuất", st.session_state["resume_text"], height=200)