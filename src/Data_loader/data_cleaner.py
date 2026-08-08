import os
import re
import json
from pdf_reader import extract_text_pdf

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def clean_resume(Input_dir, Output_dir):

    if not Input_dir:
        print("Error")
        return ""

    raw_pdf = [f for f in os.listdir(Input_dir) if f.lower().endswith('.pdf')]
    raw_pdf.sort()

    cleaned_res = []

    for idx, res in enumerate(raw_pdf, start=1):
        pdf_path = os.path.join(Input_dir, res)
        raw_text = extract_text_pdf(pdf_path)

        cleaned_text = clean_text(raw_text)
        cv_id = f"cv_{idx:03d}"

        cleaned_res.append({
            "id": cv_id,
            "filename": res,
            "text": cleaned_text
        })
    
    with open(Output_dir, 'w', encoding='utf-8') as f:
        json.dump(cleaned_res, f, indent=2, ensure_ascii=True)

raw = "../../Data/Raw/Resumes_PDF/ACCOUNTANT/"
out = "../../Data/Processed/cleaned_resumes.json"
clean_resume(raw, out)
