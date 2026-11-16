import base64
import streamlit as st
import streamlit.components.v1 as components
import fitz


@st.dialog("Xem tài liệu")
def show_document_preview(filename: str, content: bytes):
    mime = "application/pdf" if filename.lower().endswith(".pdf") else "image/*"
    encoded = base64.b64encode(content).decode("ascii")
    if mime == "application/pdf":
        # Chromium may block PDF data-URI plugins inside Streamlit's sandbox.
        # Rasterize original bytes at readable resolution instead.
        pdf = fitz.open(stream=content, filetype="pdf")
        for page in pdf:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
            st.image(pix.tobytes("png"), use_container_width=True)
        pdf.close()
    else:
        components.html(f'<img src="data:{mime};base64,{encoded}" style="max-width:100%;max-height:780px">', height=800)
