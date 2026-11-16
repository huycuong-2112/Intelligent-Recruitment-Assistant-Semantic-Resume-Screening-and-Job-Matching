import plotly.graph_objects as go
import streamlit as st
def render_radar_chart(scores: dict, title: str = "Độ phù hợp theo 4 khía cạnh"):
    """
    Vẽ radar chart thể hiện độ phù hợp CV-JD theo 4 khía cạnh:
    - Skill
    - Experience
    - Education
    - Semantic

    scores: dict dạng {"Skill": 0.8, "Experience": 0.6, "Education": 0.9, "Semantic": 0.7}
    Giá trị nằm trong khoảng 0-1.
    """
    categories = list(scores.keys())
    values = list(scores.values())

    # Nối điểm đầu vào cuối để khép kín hình radar
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Điểm số ứng viên',
        line_color='#1f77b4'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title=title
    )

    st.plotly_chart(fig, use_container_width=True)


def render_ablation_chart(ablation_results: dict, title: str = "Ablation Impact: Ảnh hưởng của từng thành phần"):
    """
    Vẽ bar chart so sánh độ chính xác khi bỏ từng thành phần ra khỏi model.

    ablation_results: dict dạng
    {
        "Full Model": 0.85,
        "Bỏ Skill Extraction": 0.72,
        "Bỏ Experience Analysis": 0.78,
        "Bỏ Semantic Embedding": 0.60
    }
    """
    labels = list(ablation_results.keys())
    values = list(ablation_results.values())

    # Tô màu khác cho "Full Model" để làm nổi bật baseline
    colors = ['#2ca02c' if label == "Full Model" else '#d62728' for label in labels]

    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.2f}" for v in values], textposition='auto')
    ])

    fig.update_layout(
        title=title,
        yaxis=dict(title="Độ chính xác (Accuracy)", range=[0, 1]),
        xaxis=dict(title="Cấu hình model")
    )

    st.plotly_chart(fig, use_container_width=True)


def render_industry_weight_comparison(industry_weights: dict, title: str = "So sánh trọng số theo ngành"):
    """
    Vẽ grouped bar chart so sánh trọng số các khía cạnh giữa nhiều ngành.

    industry_weights: dict dạng
    {
        "IT": {"Kỹ năng": 0.5, "Kinh nghiệm": 0.2, "Học vấn": 0.2, "Ngữ nghĩa": 0.1},
        "Sales": {"Kỹ năng": 0.2, "Kinh nghiệm": 0.5, "Học vấn": 0.1, "Ngữ nghĩa": 0.2}
    }
    """
    industries = list(industry_weights.keys())
    aspects = list(next(iter(industry_weights.values())).keys())

    fig = go.Figure()
    for aspect in aspects:
        values = [industry_weights[ind][aspect] for ind in industries]
        fig.add_trace(go.Bar(name=aspect, x=industries, y=values))

    fig.update_layout(
        title=title,
        barmode='group',
        yaxis=dict(title="Trọng số"),
        xaxis=dict(title="Ngành")
    )

    st.plotly_chart(fig, use_container_width=True)
