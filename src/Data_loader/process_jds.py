import pandas as pd
import json
import re
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

LINKEDIN_DIR = project_root / "Data" / "Raw" / "JD" / "Linkedin"

POSTINGS_FILE = LINKEDIN_DIR / "postings.csv"
JOB_SKILLS_FILE = LINKEDIN_DIR / "jobs" / "job_skills.csv"
JOB_INDUSTRIES_FILE = LINKEDIN_DIR / "jobs" / "job_industries.csv"
SKILLS_FILE = LINKEDIN_DIR / "mappings" / "skills.csv"
INDUSTRIES_FILE = LINKEDIN_DIR / "mappings" / "industries.csv"

OUTPUT_JSON = project_root / "Data" / "Processed" / "cleaned_jds.json"

INDUSTRY_TO_DOMAIN = {
    # 1. IT
    "Software Development": "IT",
    "Computer Networking Products": "IT",
    "Technology, Information and Internet": "IT",
    "Information Services": "IT",
    "Data Infrastructure and Analytics": "IT",
    "Computer and Network Security": "IT",

    # 2. Engineer
    "Computer Hardware Manufacturing": "Engineer",
    "Semiconductor Manufacturing": "Engineer",
    "Defense and Space Manufacturing": "Engineer",
    "Computers and Electronics Manufacturing": "Engineer",
    "Mechanical or Industrial Engineering": "Engineer",
    "Industrial Machinery Manufacturing": "Engineer",
    "Automation Machinery Manufacturing": "Engineer",
    "Aviation and Aerospace Component Manufacturing": "Engineer",
    "Electrical Equipment Manufacturing": "Engineer",

    # 3. Economics
    "Business Consulting and Services": "Economics",
    "Financial Services": "Economics",
    "Banking": "Economics",
    "Accounting": "Economics",
    "Advertising Services": "Economics",
    "Marketing Services": "Economics",
    "Real Estate": "Economics",
    "Investment Banking": "Economics",
    "Human Resources Services": "Economics",
    "Retail": "Economics",
    "Retail Apparel and Fashion": "Economics",
    "Retail Groceries": "Economics",

    # 4. Healthcare
    "Hospitals and Health Care": "Healthcare",
    "Medical Practices": "Healthcare",
    "Pharmaceutical Manufacturing": "Healthcare",
    "Medical Equipment Manufacturing": "Healthcare",
    "Biotechnology Research": "Healthcare",
    "Veterinary Services": "Healthcare",
    "Nursing Care Facilities": "Healthcare"
}

def extract_jd_experience(level_str: str, description: str) -> float:
    lvl = str(level_str).lower()
    
    # 1. Kiểm tra qua Metadata
    if 'internship' in lvl: return 0.0
    if 'entry' in lvl: return 1.0
    if 'associate' in lvl: return 2.0
    if 'mid-senior' in lvl: return 4.0
    if 'director' in lvl: return 7.0
    if 'executive' in lvl: return 10.0

    text = str(description).lower()

    # Bảng quy đổi chữ số tiếng Anh sang float
    WORD_TO_NUM = {
        'one': 1.0, 'two': 2.0, 'three': 3.0, 'four': 4.0, 'five': 5.0,
        'six': 6.0, 'seven': 7.0, 'eight': 8.0, 'nine': 9.0, 'ten': 10.0
    }

    # 2. Pattern dạng số: "2-4 years", "3+ yrs", "at least 5 years", "minimum 3 yr"
    # Match dạng khoảng (range): 2-5 years -> lấy 2.0
    range_num = re.search(r'(\d+)\s*[-–—to]+\s*(\d+)\s*(?:years?|yrs?|yr\.?)', text)
    if range_num:
        val = float(range_num.group(1))
        if 0 < val <= 15: return val

    # Match số đơn lẻ: 3+ years, 5 years of experience, min 2 yrs
    single_num = re.search(r'(?:minimum|at least|requir(?:e|ed|es)|with|have)?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?|yr\.?)\s*(?:of\s*)?(?:relevant\s*|work\s*|industry\s*|professional\s*)?(?:experience|exp)?', text)
    if single_num and single_num.group(1):
        val = float(single_num.group(1))
        if 0 < val <= 15: return val

    # 3. Pattern dạng chữ: "three to five years", "minimum two years"
    words_pattern = r'(one|two|three|four|five|six|seven|eight|nine|ten)'
    range_word = re.search(rf'{words_pattern}\s*[-–—to]+\s*{words_pattern}\s*(?:years?|yrs?|yr\.?)', text)
    if range_word:
        return WORD_TO_NUM.get(range_word.group(1), 0.0)

    single_word = re.search(rf'(?:minimum|at least|requir(?:e|ed|es)|with)\s*{words_pattern}\s*\+?\s*(?:years?|yrs?|yr\.?)', text)
    if single_word:
        return WORD_TO_NUM.get(single_word.group(1), 0.0)

    # 4. Kiểm tra chức danh nếu có chữ Senior/Lead/Junior
    title_lower = str(text[:100]).lower()
    if 'senior' in title_lower or 'sr.' in title_lower or 'lead' in title_lower:
        return 4.0
    if 'junior' in title_lower or 'jr.' in title_lower:
        return 1.0

    return 0.0

def fallback_classify_domain(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    if re.search(r'\b(software|developer|frontend|backend|fullstack|data engineer|data scientist|python|devops|cloud|ai)\b', text):
        return "IT"
    if re.search(r'\b(mechanical|mechatronics|electrical|embedded|cad|solidworks|automation|robotics|firmware|hardware|plc)\b', text):
        return "Engineer"
    if re.search(r'\b(finance|accounting|accountant|marketing|sales|business analyst|auditor|banking|hr|management)\b', text):
        return "Economics"
    if re.search(r'\b(medical|nurse|doctor|pharmacist|clinical|healthcare|biomedical|hospital)\b', text):
        return "Healthcare"
    return "Other"

def main():
    print("=" * 80)
    print("XỬ LÝ DỮ LIỆU JD (NÂNG CẤP BÓC TÁCH KINH NGHIỆM TỪ TEXT)")
    print("=" * 80)

    # 1. Bóc tách và map kỹ năng (Skills)
    print(f"⏳ 1. Đang nạp Skills từ: {JOB_SKILLS_FILE.name} & {SKILLS_FILE.name}...")
    df_job_skills = pd.read_csv(JOB_SKILLS_FILE)
    df_skills = pd.read_csv(SKILLS_FILE)
    
    df_skills_merged = pd.merge(df_job_skills, df_skills, on='skill_abr', how='left')
    skills_agg = df_skills_merged.groupby('job_id')['skill_name'].apply(lambda s: list(set(s.dropna()))).reset_index()
    skills_agg.rename(columns={'skill_name': 'skills'}, inplace=True)

    # 2. Bóc tách và map ngành nghề (Domain)
    print(f"⏳ 2. Đang nạp Industries từ: {JOB_INDUSTRIES_FILE.name} & {INDUSTRIES_FILE.name}...")
    df_job_ind = pd.read_csv(JOB_INDUSTRIES_FILE)
    df_ind = pd.read_csv(INDUSTRIES_FILE)
    
    df_ind_merged = pd.merge(df_job_ind, df_ind, on='industry_id', how='left')
    df_ind_merged['domain'] = df_ind_merged['industry_name'].map(INDUSTRY_TO_DOMAIN)
    domain_agg = df_ind_merged.dropna(subset=['domain']).groupby('job_id')['domain'].first().reset_index()

    # 3. Đọc postings.csv và gộp các bảng
    print(f"⏳ 3. Đang nạp postings từ: {POSTINGS_FILE.name}...")
    cols_to_use = ['job_id', 'title', 'description', 'formatted_experience_level']
    df_postings = pd.read_csv(POSTINGS_FILE, usecols=cols_to_use)

    df_final = pd.merge(df_postings, skills_agg, on='job_id', how='left')
    df_final = pd.merge(df_final, domain_agg, on='job_id', how='left')

    # Fallback domain nếu trống
    df_final['domain'] = df_final.apply(
        lambda row: row['domain'] if pd.notna(row['domain']) 
        else fallback_classify_domain(str(row['title']), str(row['description'])),
        axis=1
    )

    # Lọc giữ lại 4 Domain
    df_final = df_final[df_final['domain'] != "Other"].copy()

    # Áp dụng hàm trích xuất kinh nghiệm nâng cao
    print("⏳ 4. Đang quét số năm kinh nghiệm từ Metadata & Description...")
    df_final['experience_years'] = df_final.apply(
        lambda row: extract_jd_experience(row.get('formatted_experience_level', ''), row.get('description', '')),
        axis=1
    )

    df_final['skills'] = df_final['skills'].apply(lambda x: x if isinstance(x, list) else [])
    df_final.rename(columns={'title': 'job_title'}, inplace=True)
    df_final.drop(columns=['formatted_experience_level'], inplace=True, errors='ignore')
    df_final.fillna("", inplace=True)

    # 4. Xuất JSON
    print("💾 5. Đang xuất file JSON...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    records = df_final.to_dict(orient='records')
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("HOÀN TẤT XỬ LÝ JD")
    print(f"Tổng số JD giữ lại: {len(records)}")
    print("Thống kê số năm kinh nghiệm (Years of Experience):")
    print(df_final['experience_years'].value_counts().head(10).to_string())
    print(f"File lưu tại: {OUTPUT_JSON}")
    print("=" * 80)

if __name__ == "__main__":
    main()