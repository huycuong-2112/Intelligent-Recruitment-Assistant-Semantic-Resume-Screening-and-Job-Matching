import pdfplumber

sample = "../../Data/Raw/Resumes_PDF/ACCOUNTANT/10554236.pdf"

def extract_text_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"cannot process page {pdf_path}: {e}")
        return ""

    return text.strip()

print(extract_text_pdf(sample))

