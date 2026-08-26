import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import io
import streamlit as st
import fitz
from pypdf import PdfReader
from docx import Document
from app.frontend.utils.mock_extraction import mock_extract_features, CATEGORIES
from app.frontend.components.navbar import render_navbar

render_navbar("Upload CV")

st.title("📄 Upload CV")

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

if "cv_uploaded_files" not in st.session_state:
    st.session_state["cv_uploaded_files"] = []

if "cv_candidates" not in st.session_state:
    st.session_state["cv_candidates"] = None

if "cv_confirmed_candidates" not in st.session_state:
    st.session_state["cv_confirmed_candidates"] = None

# =========================
# Upload
# =========================

uploaded_files = st.file_uploader(
    "Chọn 1 hoặc nhiều file CV (PDF/DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key="cv_uploader",
)

if uploaded_files:
    existing_names = {
        f.name for f in st.session_state["cv_uploaded_files"]
    }

    for file in uploaded_files:
        if file.name not in existing_names:
            st.session_state["cv_uploaded_files"].append(file)

    st.caption(f"Đã chọn {len(uploaded_files)} file.")

# =========================
# Current files
# =========================

if st.session_state["cv_uploaded_files"]:
    files_to_remove = []

    for i, file in enumerate(st.session_state["cv_uploaded_files"]):
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
            with st.container(key=f"cv-delete-{i}"):
                st.markdown(
                    f"""
                    <style>
                    .st-key-cv-delete-{i} button:hover {{
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
                    key=f"cv_remove_{i}",
                    help=f"Xóa {file.name}"
                ):
                    files_to_remove.append(file.name)

    if files_to_remove:
        st.session_state["cv_uploaded_files"] = [
            f for f in st.session_state["cv_uploaded_files"]
            if f.name not in files_to_remove
        ]

        st.session_state["cv_candidates"] = [
            c for c in (st.session_state.get("cv_candidates") or [])
            if c["filename"] not in files_to_remove
        ]

        if not st.session_state["cv_candidates"]:
            st.session_state["cv_candidates"] = None
            st.session_state["cv_confirmed_candidates"] = None

        st.session_state["rating_results"] = None
        st.rerun()

# =========================
# Extract
# =========================

if st.session_state["cv_uploaded_files"]:
    if st.button("🔍 Extract", type="primary"):
        candidates = []

        for file in st.session_state["cv_uploaded_files"]:
            raw_features = mock_extract_features([file.name])
            candidates.append({
                "filename": file.name,
                "raw_features": raw_features
            })

        st.session_state["cv_candidates"] = candidates
        st.session_state["cv_confirmed_candidates"] = None
        st.session_state["rating_results"] = None
        st.rerun()

# =========================
# Refine
# =========================

if st.session_state.get("cv_candidates"):
    st.divider()
    st.subheader("🛠️ Refine — Kiểm tra lại thông tin đã quét")
    st.caption("Bỏ tick feature quét sai. Bấm '➕ Thêm feature' để thêm feature vào đúng trường.")

    live_candidates = []

    for idx, cand in enumerate(st.session_state["cv_candidates"]):
        with st.expander(f"👤 {cand['filename']}", expanded=True):
            kept = []
            category_cols = st.columns(4)

            for cat_idx, category in enumerate(CATEGORIES):
                with category_cols[cat_idx]:
                    with st.container(border=True):
                        st.markdown(f"### {category}")

                        feats_in_cat = [
                            (j, f)
                            for j, f in enumerate(cand["raw_features"])
                            if f["category"] == category
                        ]

                        if not feats_in_cat:
                            st.caption("Chưa có feature")

                        for j, feat in feats_in_cat:
                            if st.checkbox(
                                feat["name"],
                                value=True,
                                key=f"cv_{idx}_feat_{j}"
                            ):
                                kept.append(feat)

            # =========================
            # Add feature
            # =========================

            show_form_key = f"cv_{idx}_show_add_form"

            if show_form_key not in st.session_state:
                st.session_state[show_form_key] = False

            if not st.session_state[show_form_key]:
                if st.button("➕ Thêm feature", key=f"cv_{idx}_add_btn"):
                    st.session_state[show_form_key] = True
                    st.rerun()
            else:
                with st.form(key=f"cv_{idx}_add_form", clear_on_submit=True):
                    col_cat, col_name = st.columns([1, 2])

                    with col_cat:
                        new_cat = st.selectbox("Trường", CATEGORIES, key=f"cv_{idx}_new_cat")

                    with col_name:
                        new_name = st.text_input("Tên feature", key=f"cv_{idx}_new_name")

                    submitted = st.form_submit_button("✅ Xác nhận thêm")

                    if submitted and new_name.strip():
                        st.session_state["cv_candidates"][idx]["raw_features"].append({
                            "name": new_name.strip(),
                            "category": new_cat
                        })
                        st.session_state[show_form_key] = False
                        st.rerun()

            live_candidates.append({
                "filename": cand["filename"],
                "confirmed_features": kept
            })

    # =========================
    # Confirm status
    # =========================

    st.divider()

    confirmed = st.session_state.get("cv_confirmed_candidates")
    role = st.session_state.get("user_role")

    def total_features(candidate_list):
        return sum(len(c["confirmed_features"]) for c in candidate_list)

    if confirmed is not None and confirmed == live_candidates:
        if role == "candidate":
            st.success(
                f"✅ Đã xác nhận {total_features(confirmed)} feature. Có thể chuyển sang bước tiếp theo."
            )
        else:
            st.success(
                f"✅ Đã xác nhận {len(confirmed)} ứng viên "
                f"(tổng {total_features(confirmed)} feature). Có thể chuyển sang bước tiếp theo."
            )

    elif confirmed is not None and confirmed != live_candidates:
        st.warning("⚠️ Bạn vừa thay đổi thông tin sau khi xác nhận. Bấm Confirm lại để cập nhật.")

    else:
        st.info("Chưa xác nhận. Kiểm tra kỹ danh sách feature rồi bấm Confirm.")

    # =========================
    # Confirm
    # =========================

    if st.button("✅ Confirm", type="primary"):
        st.session_state["cv_confirmed_candidates"] = live_candidates
        st.session_state["rating_results"] = None
        st.rerun()