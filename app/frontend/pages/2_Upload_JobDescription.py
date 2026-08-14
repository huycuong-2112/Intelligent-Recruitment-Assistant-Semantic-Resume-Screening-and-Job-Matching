import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
from app.frontend.utils.api_client import call_matching_api

st.title("💼 Upload Job Description")

job_text = st.text_area("Dán nội dung mô tả công việc vào đây", height=250)

if st.button("Lưu Job Description"):
    st.session_state["job_description"] = job_text
    st.success("Đã lưu Job Description!")