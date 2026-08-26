import streamlit as st

# =========================
# Navigation items
# =========================

NAV_ITEMS = {
    "candidate": [
        ("Upload CV", "pages/1_Upload_CV.py"),
        ("Upload JD", "pages/2_Upload_JobDescription.py"),
        ("Rating", "pages/3_Matching_Result.py"),
    ],
    "hr": [
        ("Upload CV", "pages/1_Upload_CV.py"),
        ("Upload JD", "pages/2_Upload_JobDescription.py"),
        ("Rating", "pages/3_Matching_Result.py"),
        ("Ranking", "pages/4_Candidate_Ranking.py"),
    ],
}

RESETTABLE_PREFIXES = ("cv_", "jd_", "resume_", "job_", "ranking_")

# =========================
# CSS
# =========================

HIDE_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}
[data-testid="stHeader"] {display: none;}
.block-container {padding-top: 0 !important;}

.navbar-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: -1.2rem;
    margin-bottom: 0.5rem;
}

.navbar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: white;
}

.navbar-home a {
    color: white !important;
    text-decoration: none !important;
    font-weight: 500;
}

.navbar-role button {
    background: #262730 !important;
    color: white !important;
    border: 1px solid #3d3d4d !important;
    border-radius: 6px !important;
}

.navbar-role button:hover {
    background: #333440 !important;
}

/* =========================
   TAB CONTAINER
   ========================= */

.st-key-nav-tabs [data-testid="stHorizontalBlock"] {
    display: flex !important;
    gap: 0 !important;
    width: 100% !important;
    align-items: stretch !important;
}

.st-key-nav-tabs [data-testid="stColumn"] {
    flex: 1 1 0 !important;
    width: 0 !important;
    min-width: 0 !important;
    padding: 0 !important;
    transition: flex 0.2s ease;
}

/* =========================
   TAB BUTTON
   ========================= */

.st-key-nav-tabs button {
    width: 100% !important;
    min-height: 2.8rem !important;
    padding: 0.55rem 0.5rem !important;

    background: #262730 !important;
    color: white !important;

    border: 1px solid #3d3d4d !important;
    border-right: none !important;
    border-radius: 8px 8px 0 0 !important;

    font-weight: 500 !important;

    transition:
        background-color 0.2s ease,
        font-weight 0.2s ease;
}

.st-key-nav-tabs [data-testid="stColumn"]:last-child button {
    border-right: 1px solid #3d3d4d !important;
}

.st-key-nav-tabs button:hover {
    background: #333440 !important;
    color: white !important;
    font-weight: 700 !important;
}

/* =========================
   HOVER EXPAND
   ========================= */

.st-key-nav-tabs [data-testid="stColumn"]:has(button:hover) {
    flex: 1.6 1 0 !important;
}

.st-key-nav-tabs [data-testid="stColumn"]:not(:has(button:hover)) {
    flex: 0.85 1 0 !important;
}
</style>
"""


# =========================
# Navbar
# =========================

def render_navbar(current_label: str, hr_only: bool = False):
    role = st.session_state.get("user_role")

    # =========================
    # Role check
    # =========================

    if role is None:
        st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)
        st.warning("Vui lòng chọn vai trò ở trang Home trước.")
        st.page_link("Home.py", label="⬅️ Về trang Home")
        st.stop()

    if hr_only and role != "hr":
        st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)
        st.error("Mục này chỉ dành cho HR.")
        st.stop()

    items = NAV_ITEMS[role]

    st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

    # =========================
    # Header
    # =========================

    st.markdown(
        '<div class="navbar-header"><div class="navbar-title">Intelligent Recruitment Assistant</div></div>',
        unsafe_allow_html=True
    )

    # =========================
    # Home + đổi role
    # =========================

    col_home, col_space, col_role = st.columns([5, 3, 2], gap=None)

    with col_home:
        st.markdown('<div class="navbar-home">', unsafe_allow_html=True)
        st.page_link("Home.py", label="🏠 Home")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_role:
        st.markdown('<div class="navbar-role">', unsafe_allow_html=True)

        if st.button(
            "🔄 Đổi vai trò",
            key="navbar_switch_role",
            use_container_width=True
        ):
            for key in list(st.session_state.keys()):
                if key.startswith(RESETTABLE_PREFIXES):
                    del st.session_state[key]

            st.session_state["user_role"] = None
            st.switch_page("Home.py")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    # =========================
    # Tabs
    # =========================

    with st.container(key="nav-tabs"):
        columns = st.columns(len(items), gap=None)

        for i, (label, path) in enumerate(items):
            with columns[i]:
                with st.container(key=f"nav-tab-{i}"):
                    if label == current_label:
                        st.markdown(
                            f"""
                            <style>
                            .st-key-nav-tab-{i} button {{
                                background: #0e1117 !important;
                                color: white !important;
                                font-weight: 700 !important;
                                border-bottom: 2px solid white !important;
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

                    if st.button(
                        label,
                        key=f"nav_button_{i}",
                        use_container_width=True
                    ):
                        st.switch_page(path)

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)