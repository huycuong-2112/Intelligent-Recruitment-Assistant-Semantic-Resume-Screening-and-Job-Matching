import streamlit as st

st.set_page_config(
    page_title="Intelligent Recruitment Assistant",
    page_icon="🧑‍💼",
    layout="wide"
)

st.title("🧑‍💼 Intelligent Recruitment Assistant")
st.markdown("""
Hệ thống hỗ trợ tuyển dụng thông minh: sàng lọc CV và so khớp mô tả công việc
bằng mô hình Transformer.

**Điều hướng:** dùng menu bên trái để:
- Upload CV
- Upload Job Description
- Xem kết quả Matching
""")