import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import io
import streamlit as st
import fitz
from app.frontend.utils.api_client import upload_resume, submit_resume_extraction, get_extraction_status, confirm_resume, ResumeParseAPIError
from app.frontend.utils.state_utils import feature_selection_fingerprint, confirmation_freshness, is_runtime_ready
from app.frontend.components.navbar import render_navbar
from app.frontend.utils.manual_feature_contract import TYPE_OPTIONS, labels_for, subtype_selector_visible
from app.frontend.utils.document_preview import show_document_preview
from app.frontend.utils.extraction_jobs import collect as collect_extraction_jobs, public_job, start_jobs

CATEGORIES = ["Education", "Skills", "Experience", "Projects"]

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

def get_preview(file):
    suffix = os.path.splitext(file.name)[1].lower()
    if suffix == ".pdf":
        return get_pdf_thumbnail(file)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return file.getvalue()
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
if "cv_extraction_jobs" not in st.session_state:
    st.session_state["cv_extraction_jobs"] = {}
if "cv_uploader_epoch" not in st.session_state:
    st.session_state["cv_uploader_epoch"] = 0

def _tombstone_cv_jobs(filenames):
    """Cancel queue ownership for files removed while a batch is active."""
    jobs = st.session_state.get("cv_extraction_jobs")
    if not isinstance(jobs, dict):
        return
    for job in jobs.values():
        if isinstance(job, dict) and job.get("filename") in filenames and job.get("status") in {"pending", "queued", "running", "processing", "long_running"}:
            job["status"] = "removed"

# =========================
# Upload
# =========================

def _native_cv_removed():
    current = {f.name for f in (st.session_state.get(f"cv_uploader_{st.session_state['cv_uploader_epoch']}") or [])}
    active = st.session_state.get("cv_uploaded_files", [])
    removed = {f.name for f in active} - current
    if removed:
        _tombstone_cv_jobs(removed)
        st.session_state["cv_uploaded_files"] = [f for f in active if f.name not in removed]
        st.session_state["cv_candidates"] = [c for c in (st.session_state.get("cv_candidates") or []) if c.get("filename") not in removed] or None
        st.session_state["cv_confirmed_candidates"] = [c for c in (st.session_state.get("cv_confirmed_candidates") or []) if c.get("filename") not in removed] or None
        st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None

uploaded_files = st.file_uploader(
    "Chọn 1 hoặc nhiều file CV (PDF/PNG/JPG/JPEG)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"cv_uploader_{st.session_state['cv_uploader_epoch']}",
    on_change=_native_cv_removed,
)

# Reconcile native uploader removals with application-owned state.
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
            thumbnail = get_preview(file)

            if thumbnail:
                st.image(thumbnail, width=90)
                if st.button("Xem", key=f"cv_preview_{i}"):
                    show_document_preview(file.name, file.getvalue())
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
        _tombstone_cv_jobs(set(files_to_remove))
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
        st.session_state["ranking_df"] = None
        st.session_state["cv_uploader_epoch"] += 1
        st.rerun()

# =========================
# Extract
# =========================

if st.session_state["cv_uploaded_files"]:
    if st.button("🔍 Extract", type="primary"):
        st.session_state["cv_candidates"] = []
        st.session_state["cv_extraction_jobs"], _ = start_jobs(st.session_state["cv_uploaded_files"], upload_resume)
        st.session_state["cv_confirmed_candidates"] = None
        st.session_state["rating_results"] = None
        st.session_state["ranking_df"] = None
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
            extraction = cand.get("extraction") or {}
            source_status = extraction.get("status")
            extraction_method = (cand.get("parsed") or {}).get("extraction_method")
            if source_status == "LOW_QUALITY": st.warning("⚠ Chất lượng quét/trích xuất: Thấp — Nên kiểm tra kỹ thông tin trong bước Refine.")
            elif source_status == "RECOVERED_BY_OCR": st.info("ℹ️ Đã khôi phục bằng OCR; vui lòng kiểm tra Refine.")
            elif source_status == "ACCEPTED_BY_DOCLING": st.success("✅ Chất lượng trích xuất: Tốt")
            elif source_status: st.caption(f"Source status: {source_status}")
            if extraction_method: st.caption(f"Phương thức trích xuất: {extraction_method}")
            if extraction_method == "offline_hybrid": st.caption("ℹ️ CV được xử lý bằng fallback offline; scoring semantics không thay đổi.")
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
                # Keep these widgets outside a form so changing category
                # immediately reruns Streamlit and updates subtype choices.
                new_cat = st.selectbox("Trường", list(TYPE_OPTIONS), key=f"cv_{idx}_new_cat")
                if subtype_selector_visible(new_cat):
                    selected_type = st.selectbox("Loại", labels_for(new_cat), key=f"cv_{idx}_new_type_{new_cat}")
                else:
                    selected_type = labels_for(new_cat)[0]
                new_name = st.text_input("Giá trị", key=f"cv_{idx}_new_name")
                submitted = st.button("✅ Xác nhận thêm", key=f"cv_{idx}_add_submit")

                if submitted and new_name.strip():
                        feature_type = next(x[0] for x in TYPE_OPTIONS[new_cat] if x[1] == selected_type)
                        existing = st.session_state["cv_candidates"][idx]["raw_features"]
                        from src.Normalization.skill_normalizer import normalize_skill
                        key = normalize_skill(new_name).casefold() if new_cat == "Skills" else " ".join(new_name.strip().casefold().split())
                        duplicate = any((normalize_skill(f.get("name", "")).casefold() if new_cat == "Skills" else " ".join(str(f.get("name", "")).strip().casefold().split())) == key and f.get("category") == new_cat for f in existing)
                        if duplicate:
                            st.error(f"{new_name.strip()} đã tồn tại trong {new_cat}.")
                        else:
                            existing.append({"name": new_name.strip(), "category": new_cat, "feature_type": feature_type, "source_type": "manual_ui"})
                            st.session_state[show_form_key] = False
                            st.rerun()

            live_candidates.append({
                "filename": cand["filename"],
                "run_id": cand.get("run_id"),
                "document_id": cand.get("document_id"),
                "extraction": cand.get("extraction", {}),
                "parsed": cand.get("parsed"),
                "confirmed_features": kept
            })

    # Freshness is tracked per document, independently of list ordering.
    confirmed_by_id = {c.get("document_id"): c for c in (st.session_state.get("cv_confirmed_candidates") or [])}
    for live in live_candidates:
        live["current_feature_fingerprint"] = feature_selection_fingerprint(live["confirmed_features"])
        prior = confirmed_by_id.get(live.get("document_id"))
        live["confirmation_freshness"] = confirmation_freshness(prior or {}, live["confirmed_features"])
        live["confirmation_dirty"] = live["confirmation_freshness"] == "DIRTY"
    draft_key = {x.get("document_id"): x.get("current_feature_fingerprint") for x in live_candidates}
    if draft_key != st.session_state.get("_cv_last_draft_fingerprints"):
        old_key = st.session_state.get("_cv_last_draft_fingerprints")
        if old_key is not None and draft_key != old_key:
            st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None
        st.session_state["_cv_last_draft_fingerprints"] = draft_key

    # =========================
    # Confirm status
    # =========================

    st.divider()

    confirmed = st.session_state.get("cv_confirmed_candidates")
    role = st.session_state.get("user_role")

    def total_features(candidate_list):
        return sum(len(c["confirmed_features"]) for c in candidate_list)

    if confirmed is not None and all(not c.get("confirmation_dirty") and c.get("confirmation_freshness") == "CONFIRMED" for c in live_candidates):
        if role == "candidate":
            st.success(
                f"✅ Đã xác nhận {total_features(confirmed)} feature. Có thể chuyển sang bước tiếp theo."
            )
        else:
            st.success(
                f"✅ Đã xác nhận {len(confirmed)} ứng viên "
                f"(tổng {total_features(confirmed)} feature). Có thể chuyển sang bước tiếp theo."
            )

    elif confirmed is not None:
        st.warning("⚠️ Bạn vừa thay đổi thông tin sau khi xác nhận. Bấm Confirm lại để cập nhật.")

    else:
        st.info("Chưa xác nhận. Kiểm tra kỹ danh sách feature rồi bấm Confirm.")

    # =========================
    # Confirm
    # =========================

    if st.button("✅ Confirm", type="primary"):
        confirmed_results = []
        previous = {c.get("document_id"): c for c in (st.session_state.get("cv_confirmed_candidates") or [])}
        for cand, live in zip(st.session_state["cv_candidates"], live_candidates):
            try:
                result = confirm_resume(cand["run_id"], cand["document_id"], live["confirmed_features"])
                confirmed_results.append({**live, "confirm_status": result["status"], "override": result["override"], "runtime_parsed": result["runtime_parsed"], "applied_actions": result.get("applied_actions", []), "unsupported_actions": result.get("unsupported_actions", []), "confirmed_feature_fingerprint": feature_selection_fingerprint(live["confirmed_features"]), "confirmation_dirty": False, "confirmation_freshness": "CONFIRMED"})
                if result["status"] == "PARTIAL": st.warning(f"{cand['filename']}: some changes were not applied.")
                else: st.success(f"{cand['filename']}: confirmation applied.")
            except ResumeParseAPIError as exc:
                st.error(f"{cand['filename']}: {exc}")
                if cand.get("document_id") in previous:
                    confirmed_results.append(previous[cand["document_id"]])
        if confirmed_results:
            st.session_state["cv_confirmed_candidates"] = confirmed_results
            st.session_state["rating_results"] = None
            st.session_state["ranking_df"] = None
        st.rerun()

if st.session_state.get("cv_extraction_jobs"):
    @st.fragment(run_every="1s")
    def _poll_cv_jobs():
        jobs = st.session_state["cv_extraction_jobs"]
        if not isinstance(jobs, dict) or not jobs:
            st.session_state["cv_extraction_jobs"] = {}
            return
        changed = collect_extraction_jobs(jobs, (submit_resume_extraction, lambda job_id: get_extraction_status("resume", job_id)))
        candidates = {c.get("filename"): c for c in (st.session_state.get("cv_candidates") or [])}
        valid_jobs = [job for job in jobs.values() if isinstance(job, dict)]
        for job in valid_jobs:
            if job.get("status") == "completed" and job.get("result") is not None and job["filename"] not in candidates:
                response = job["result"]
                candidates[job["filename"]] = {"filename": response.get("filename", job["filename"]), "run_id": response["run_id"], "document_id": response["document_id"], "extraction": response.get("extraction", {}), "parsed": response["parsed"], "raw_features": response.get("ui_features", [])}
        st.session_state["cv_candidates"] = [candidates[k] for k in sorted(candidates, key=lambda n: next((j["order"] for j in valid_jobs if j.get("filename") == n), 0))]
        for job in sorted(valid_jobs, key=lambda x: x.get("order", 0)):
            if job["status"] == "pending": st.caption(f"{job['filename']}: Pending")
            elif job["status"] in {"queued", "running", "processing"}: st.caption(f"{job['filename']}: Đang trích xuất...")
            elif job["status"] == "long_running": st.info(f"{job['filename']}: Đang xử lý lâu hơn dự kiến... (job vẫn được theo dõi)")
            elif job["status"] == "completed": st.caption(f"{job['filename']}: Sẵn sàng")
            elif job["status"] == "failed": st.error(f"{job['filename']}: Lỗi — {job.get('error', 'Trích xuất thất bại')}")
            else: st.caption(f"{job['filename']}: Đang chờ trạng thái...")
        if valid_jobs and all(j.get("status") in {"completed", "failed", "removed"} for j in valid_jobs):
            # Keep the queue as a valid inactive mapping.  A fragment can
            # execute once more after this state transition.
            st.session_state["cv_extraction_jobs"] = {}
        elif changed: st.rerun(scope="app")
    _poll_cv_jobs()
