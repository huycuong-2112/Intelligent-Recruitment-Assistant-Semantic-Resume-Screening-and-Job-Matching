import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import io
import streamlit as st
import fitz
from pypdf import PdfReader
from docx import Document
from app.frontend.utils.mock_extraction import mock_extract_features, CATEGORIES
from app.frontend.components.navbar import render_navbar

render_navbar("Upload JD")

st.title("💼 Upload Job Description")

# =========================
# Helper
# =========================

def get_pdf_thumbnail(file, width=110):
    try:
        pdf = fitz.open(stream=file.getvalue(), filetype="pdf")

        if len(pdf) == 0:
            return None

        page = pdf[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)

        image = pix.tobytes("png")
        pdf.close()

        return image

    except Exception:
        return None


# =========================
# Session state
# =========================

if "jd_uploaded_files" not in st.session_state:
    st.session_state["jd_uploaded_files"] = []

if "jd_features" not in st.session_state:
    st.session_state["jd_features"] = None

if "jd_confirmed_features" not in st.session_state:
    st.session_state["jd_confirmed_features"] = None

# =========================
# Upload
# =========================

uploaded_files = st.file_uploader(
    "Chọn 1 hoặc nhiều file JD (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="jd_uploader",
)

if uploaded_files:
    existing_names = {
        f.name for f in st.session_state["jd_uploaded_files"]
    }

    for file in uploaded_files:
        if file.name not in existing_names:
            st.session_state["jd_uploaded_files"].append(file)

    st.caption(f"Đã chọn {len(uploaded_files)} file.")

# =========================
# Current files
# =========================

if st.session_state["jd_uploaded_files"]:
    files_to_remove = []

    for i, file in enumerate(st.session_state["jd_uploaded_files"]):
        col_preview, col_info, col_remove = st.columns([1.2, 7, 0.8])

        with col_preview:
            thumbnail = get_pdf_thumbnail(file)

            if thumbnail:
                st.image(thumbnail, width=90)
            else:
                st.markdown("📄")

        with col_info:
            st.markdown(f"**{file.name}**")
            st.caption(f"{file.size / (1024 * 1024):.2f} MB")

        with col_remove:
            with st.container(key=f"jd-delete-{i}"):
                st.markdown(
                    f"""
                    <style>
                    .st-key-jd-delete-{i} button:hover {{
                        background: #d32f2f !important;
                        border-color: #ff5252 !important;
                        color: white !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "🗑️",
                    key=f"jd_remove_{i}",
                    help=f"Xóa {file.name}"
                ):
                    files_to_remove.append(file.name)

    if files_to_remove:
        st.session_state["jd_uploaded_files"] = [
            f for f in st.session_state["jd_uploaded_files"]
            if f.name not in files_to_remove
        ]

        st.session_state["jd_candidates"] = [
            c for c in (st.session_state.get("jd_candidates") or [])
            if c["filename"] not in files_to_remove
        ]

        if not st.session_state["jd_candidates"]:
            st.session_state["jd_candidates"] = None
            st.session_state["jd_confirmed_candidates"] = None

        st.session_state["rating_results"] = None
        st.rerun()

# =========================
# Extract
# =========================

if st.session_state["jd_uploaded_files"]:
    if st.button("🔍 Extract", type="primary"):
        filenames = [
            file.name
            for file in st.session_state["jd_uploaded_files"]
        ]

        st.session_state["jd_features"] = mock_extract_features(filenames)
        st.session_state["jd_confirmed_features"] = None
        st.session_state["rating_results"] = None
        st.rerun()

# =========================
# Refine
# =========================

if st.session_state.get("jd_features"):
    st.divider()
    st.subheader("🛠️ Refine — Kiểm tra lại thông tin đã quét")
    st.caption("Bỏ tick feature quét sai. Bấm '➕ Thêm feature' để thêm feature vào đúng trường.")

    kept = []
    category_cols = st.columns(4)

    for i, category in enumerate(CATEGORIES):
        with category_cols[i]:
            with st.container(border=True):
                st.markdown(f"### {category}")

                feats_in_cat = [
                    (j, f)
                    for j, f in enumerate(st.session_state["jd_features"])
                    if f["category"] == category
                ]

                if not feats_in_cat:
                    st.caption("Chưa có feature")

                for j, feat in feats_in_cat:
                    if st.checkbox(
                        feat["name"],
                        value=True,
                        key=f"jd_feat_{j}"
                    ):
                        kept.append(feat)

    # =========================
    # Add feature
    # =========================

    show_form_key = "jd_show_add_form"

    if show_form_key not in st.session_state:
        st.session_state[show_form_key] = False

    if not st.session_state[show_form_key]:
        if st.button("➕ Thêm feature", key="jd_add_btn"):
            st.session_state[show_form_key] = True
            st.rerun()
    else:
        with st.form(key="jd_add_form", clear_on_submit=True):
            col_cat, col_name = st.columns([1, 2])

            with col_cat:
                new_cat = st.selectbox("Trường", CATEGORIES, key="jd_new_cat")

            with col_name:
                new_name = st.text_input("Tên feature", key="jd_new_name")

            submitted = st.form_submit_button("✅ Xác nhận thêm")

            if submitted and new_name.strip():
                st.session_state["jd_features"].append({
                    "name": new_name.strip(),
                    "category": new_cat
                })
                st.session_state[show_form_key] = False
                st.rerun()

    # =========================
    # Confirm status
    # =========================

    st.divider()

    confirmed = st.session_state.get("jd_confirmed_features")

    if confirmed is not None and confirmed == kept:
        st.success(
            f"✅ Đã xác nhận {len(confirmed)} feature. "
            "Có thể chuyển sang bước tiếp theo."
        )

    elif confirmed is not None and confirmed != kept:
        st.warning(
            "⚠️ Bạn vừa thay đổi danh sách feature sau khi xác nhận. "
            "Bấm Confirm lại để cập nhật."
        )

    else:
        st.info(
            "Chưa xác nhận. Kiểm tra kỹ danh sách feature rồi bấm Confirm."
        )

    # =========================
    # Confirm
    # =========================

    if st.button("✅ Confirm", type="primary"):
        st.session_state["jd_confirmed_features"] = kept
        st.session_state["rating_results"] = None
        st.rerun()