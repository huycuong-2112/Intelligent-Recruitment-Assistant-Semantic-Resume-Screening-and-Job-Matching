import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
from app.frontend.utils.api_client import call_matching_api

st.title("🔍 Kết quả Matching")

resume_text = st.session_state.get("resume_text")
job_description = st.session_state.get("job_description")

if not resume_text or not job_description:
    st.warning("Vui lòng upload CV và Job Description trước.")
else:
    if st.button("Tính độ tương đồng"):
        with st.spinner("Đang gọi model để tính toán..."):
            result = call_matching_api(resume_text, job_description)
        st.metric("Similarity Score", f"{result['similarity_score']*100:.1f}%")
        st.write(result.get("explanation"))