import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
from app.frontend.utils.api_client import call_matching_api
from app.frontend.components.charts import (
    render_radar_chart,
    render_ablation_chart,
    render_industry_weight_comparison,
)

st.title("📊 Analytics Dashboard")
st.markdown("Trang demo các biểu đồ phân tích (đang dùng dữ liệu mẫu để test giao diện).")

st.divider()

# --- Demo 1: Radar chart 4 khía cạnh ---
st.subheader("1. Radar Chart — Edu / Skill / Domain / Exp Score")
mock_radar_data = {
    "Edu Score": 0.70,
    "Skill Score": 0.85,
    "Domain Score": 0.68,
    "Exp Score": 0.60,
}
render_radar_chart(mock_radar_data)

st.divider()

# --- Demo 2: Ablation impact chart ---
st.subheader("2. Ablation Impact Chart")
mock_ablation_data = {
    "Full Model": 0.85,
    "Bỏ Edu Score": 0.79,
    "Bỏ Skill Score": 0.68,
    "Bỏ Domain Score": 0.75,
    "Bỏ Exp Score": 0.71,
}
render_ablation_chart(mock_ablation_data)

st.divider()

# --- Demo 3: So sánh trọng số theo ngành ---
st.subheader("3. So sánh trọng số theo ngành")
mock_industry_data = {
    "IT": {"Edu Score": 0.15, "Skill Score": 0.50, "Domain Score": 0.20, "Exp Score": 0.15},
    "Sales": {"Edu Score": 0.10, "Skill Score": 0.20, "Domain Score": 0.20, "Exp Score": 0.50},
    "Finance": {"Edu Score": 0.30, "Skill Score": 0.25, "Domain Score": 0.25, "Exp Score": 0.20},
}
render_industry_weight_comparison(mock_industry_data)