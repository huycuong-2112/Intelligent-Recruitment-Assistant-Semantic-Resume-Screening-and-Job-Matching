import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import streamlit as st
import fitz
from app.frontend.utils.api_client import upload_job, submit_job_extraction, get_extraction_status, confirm_job, ResumeParseAPIError
from app.frontend.utils.state_utils import feature_selection_fingerprint, confirmation_freshness
from app.frontend.components.navbar import render_navbar
from src.Normalization.skill_normalizer import normalize_skill
from app.frontend.utils.document_preview import show_document_preview

CATEGORIES = ["Education", "Required Skills", "Preferred Skills", "Responsibilities", "Certifications"]
render_navbar("Upload JD")
st.title("💼 Upload Job Description")

def preview(file):
    if file.name.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=file.getvalue(), filetype="pdf")
            image = doc[0].get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False).tobytes("png") if len(doc) else None
            doc.close(); return image
        except Exception: return None
    return file.getvalue() if os.path.splitext(file.name)[1].lower() in {".png", ".jpg", ".jpeg"} else None

for key, default in (("jd_uploaded_files", []), ("jd_document", None), ("jd_features", None), ("jd_confirmed_features", None), ("jd_confirmed_document", None), ("jd_uploader_epoch", 0), ("jd_extraction_processing", False), ("jd_extraction_job_id", None)):
    if key not in st.session_state: st.session_state[key] = default

def _native_jd_removed():
    if st.session_state.get(f"jd_uploader_{st.session_state['jd_uploader_epoch']}") is None and st.session_state.get("jd_uploaded_files"):
        st.session_state["jd_uploaded_files"] = []
        for key in ("jd_document", "jd_features", "jd_confirmed_features", "jd_confirmed_document"): st.session_state[key] = None
        st.session_state["jd_extraction_processing"] = False
        st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None

uploaded = st.file_uploader("Chọn 1 file JD (PDF/PNG/JPG/JPEG)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False, key=f"jd_uploader_{st.session_state['jd_uploader_epoch']}", on_change=_native_jd_removed)
if uploaded and (not st.session_state["jd_uploaded_files"] or st.session_state["jd_uploaded_files"][0].name != uploaded.name):
    st.session_state["jd_uploaded_files"] = [uploaded]
    for key in ("jd_document", "jd_features", "jd_confirmed_features", "jd_confirmed_document"): st.session_state[key] = None
    st.session_state["jd_extraction_processing"] = False
    st.session_state["_jd_last_draft_fingerprint"] = None
    st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None

files = st.session_state["jd_uploaded_files"]
if files:
    file = files[0]; a, b, c = st.columns([1.2, 7, .8])
    with a:
        image = preview(file)
        if image:
            st.image(image, width=90)
            if st.button("Xem", key="jd_preview"):
                show_document_preview(file.name, file.getvalue())
        else:
            st.markdown("📄")
    with b: st.markdown(f"**{file.name}**"); st.caption(f"{file.size / (1024 * 1024):.2f} MB")
    with c:
        if st.button("🗑️", key="jd_remove"):
            st.session_state["jd_uploaded_files"] = []
            st.session_state["jd_uploader_epoch"] += 1
            for key in ("jd_document", "jd_features", "jd_confirmed_features", "jd_confirmed_document"): st.session_state[key] = None
            st.session_state["jd_extraction_processing"] = False
            st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None; st.rerun()

if files and st.button("🔍 Extract", type="primary"):
    st.session_state["jd_extraction_processing"] = True
    try:
        st.session_state["jd_extraction_job_id"] = submit_job_extraction(file)["job_id"]
        st.session_state["jd_document"] = None; st.session_state["jd_features"] = None
        st.session_state["jd_confirmed_features"] = None; st.session_state["jd_confirmed_document"] = None
        st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None; st.rerun()
    except ResumeParseAPIError as exc: st.session_state["jd_extraction_processing"] = False; st.error(str(exc))

if st.session_state.get("jd_extraction_processing"):
    st.info("⏳ Đang xử lý JD…")

if st.session_state.get("jd_extraction_job_id") and not st.session_state.get("jd_document"):
    @st.fragment(run_every="1s")
    def _poll_jd_job():
        try: status = get_extraction_status("job", st.session_state["jd_extraction_job_id"])
        except ResumeParseAPIError as exc: st.error(str(exc)); return
        if status.get("status") == "completed" and status.get("result"):
            response=status["result"]; document={"filename":response.get("filename",status.get("filename","JD")),"run_id":response["run_id"],"document_id":response["document_id"],"extraction":response.get("extraction",{}),"parsed":response["parsed"],"raw_features":response.get("ui_features",[])}
            st.session_state["jd_document"]=document; st.session_state["jd_features"]=document["raw_features"]; st.session_state["jd_extraction_processing"]=False; st.session_state["jd_extraction_job_id"]=None; st.rerun(scope="app")
        elif status.get("status") == "failed": st.session_state["jd_extraction_processing"]=False; st.error(status.get("error","JD extraction failed"))
        else: st.caption(f"JD: {status.get('status','queued')}…")
    _poll_jd_job()

if st.session_state.get("jd_document"):
    extraction = st.session_state["jd_document"].get("extraction") or {}
    source_status = extraction.get("status")
    extraction_method = (st.session_state["jd_document"].get("parsed") or {}).get("extraction_method")
    if source_status == "ACCEPTED_BY_DOCLING":
        st.success("✅ Chất lượng trích xuất: Tốt")
    elif source_status == "RECOVERED_BY_OCR":
        st.info("ℹ️ Đã phục hồi bằng OCR")
    elif source_status == "LOW_QUALITY":
        st.warning("⚠️ Chất lượng trích xuất: Thấp — vui lòng kiểm tra kỹ trước khi Confirm.")
    elif source_status:
        st.caption(f"Source status: {source_status}")
    if extraction_method:
        st.caption(f"Phương thức trích xuất: {extraction_method}")
    if extraction_method == "offline_hybrid":
        st.caption("ℹ️ JD được xử lý bằng fallback offline; scoring semantics không thay đổi.")
    st.divider(); st.subheader("🛠️ Refine — Kiểm tra lại thông tin đã quét")
    kept = []; columns = st.columns(len(CATEGORIES))
    for i, category in enumerate(CATEGORIES):
        with columns[i]:
            with st.container(border=True):
                st.markdown(f"### {category}")
                items = [(j, x) for j, x in enumerate(st.session_state["jd_features"] or []) if x.get("category") == category]
                if not items: st.caption("Chưa có feature")
                for j, feature in items:
                    if st.checkbox(feature["name"], value=True, key=f"jd_feat_{j}"): kept.append(feature)
    current_fp = feature_selection_fingerprint(kept)
    prior = st.session_state.get("jd_confirmed_document") or {}
    freshness = confirmation_freshness(prior, kept)
    if current_fp != st.session_state.get("_jd_last_draft_fingerprint"):
        if st.session_state.get("_jd_last_draft_fingerprint") is not None:
            st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None
        st.session_state["_jd_last_draft_fingerprint"] = current_fp
    if "jd_show_add_form" not in st.session_state: st.session_state["jd_show_add_form"] = False
    if not st.session_state["jd_show_add_form"]:
        if st.button("➕ Thêm feature", key="jd_add_btn"): st.session_state["jd_show_add_form"] = True; st.rerun()
    else:
        type_options = {
            "Education": [("required_degree", "Required Degree"), ("preferred_field", "Preferred Field of Study")],
            "Required Skills": [("required_skill", "Required Skill")],
            "Preferred Skills": [("preferred_skill", "Preferred Skill")],
            "Responsibilities": [("responsibility", "Responsibility")],
            "Certifications": [("certification", "Certification")],
        }
        category = st.selectbox("Trường", list(type_options), key="jd_add_category")
        if len(type_options[category]) > 1:
            selected_type = st.selectbox("Loại", [label for _, label in type_options[category]], key=f"jd_add_type_{category}")
        else:
            selected_type = type_options[category][0][1]
        name = st.text_input("Giá trị", key="jd_add_value")
        if st.button("✅ Xác nhận thêm", key="jd_add_submit") and name.strip():
            feature_type = next(value for value, label in type_options[category] if label == selected_type)
            key = normalize_skill(name).casefold() if category in {"Required Skills", "Preferred Skills"} else " ".join(name.strip().casefold().split())
            duplicate = any(
                f.get("category") == category and
                (normalize_skill(f.get("name", "")).casefold() if category in {"Required Skills", "Preferred Skills"} else " ".join(str(f.get("name", "")).strip().casefold().split())) == key
                for f in (st.session_state["jd_features"] or [])
            )
            if duplicate:
                st.error(f"{name.strip()} đã tồn tại trong {category}.")
            else:
                st.session_state["jd_features"].append({"name": name.strip(), "category": category, "feature_type": feature_type, "source_type": "manual_ui"})
                st.session_state["jd_show_add_form"] = False; st.rerun()
    st.divider(); confirmed = st.session_state.get("jd_confirmed_features")
    if confirmed is not None and freshness == "CONFIRMED": st.success(f"✅ Đã xác nhận {len(confirmed)} feature.")
    elif confirmed is not None: st.warning("⚠️ Bạn vừa thay đổi danh sách feature. Bấm Confirm lại.")
    else: st.info("Chưa xác nhận. Kiểm tra danh sách rồi bấm Confirm.")
    if st.button("✅ Confirm", type="primary"):
        try:
            result = confirm_job(st.session_state["jd_document"]["run_id"], st.session_state["jd_document"]["document_id"], kept)
            st.session_state["jd_confirmed_features"] = kept
            st.session_state["jd_confirmed_document"] = {**st.session_state["jd_document"], "confirmed_features": kept, "confirm_status": result["status"], "override": result["override"], "runtime_parsed": result["runtime_parsed"], "applied_actions": result.get("applied_actions", []), "unsupported_actions": result.get("unsupported_actions", []), "confirmed_feature_fingerprint": feature_selection_fingerprint(kept), "confirmation_freshness": "CONFIRMED", "confirmation_dirty": False}
            st.session_state["rating_results"] = None; st.session_state["ranking_df"] = None
            st.warning("Một số thay đổi chưa áp dụng được.") if result["status"] == "PARTIAL" else st.success("Confirmation applied successfully.")
        except ResumeParseAPIError as exc: st.error(str(exc))
        st.rerun()
