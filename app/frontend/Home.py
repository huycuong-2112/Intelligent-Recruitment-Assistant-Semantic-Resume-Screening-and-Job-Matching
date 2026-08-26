import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from app.frontend.components.navbar import render_navbar, HIDE_SIDEBAR_CSS

st.set_page_config(
    page_title="Intelligent Recruitment Assistant",
    page_icon="🧑‍💼",
    layout="wide"
)
st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

if "user_role" not in st.session_state:
    st.session_state["user_role"] = None

st.title("🧑‍💼 Intelligent Recruitment Assistant")

if st.session_state["user_role"] is None:
    st.markdown("### Chào bạn! Bạn là ai?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👤 Candidate", use_container_width=True):
            st.session_state["user_role"] = "candidate"
            st.switch_page("pages/1_Upload_CV.py")
    with col2:
        if st.button("🧑‍💼 HR", use_container_width=True):
            st.session_state["user_role"] = "hr"
            st.switch_page("pages/1_Upload_CV.py")
else:
    render_navbar("Home")
    st.success(f"Vai trò hiện tại: **{st.session_state['user_role'].upper()}**")
    st.markdown("Chọn mục ở menu phía trên để tiếp tục.")