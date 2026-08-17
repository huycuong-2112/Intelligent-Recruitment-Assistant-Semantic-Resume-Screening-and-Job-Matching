import pandas as pd
import json
import re
from pathlib import Path

# Tìm thư mục gốc project
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

ESCO_SKILLS_CSV = project_root / "Data" / "Raw" / "ESCO" / "skills_en.csv"
OUTPUT_SUBSET_JSON = project_root / "Data" / "Processed" / "esco_subset.json"

# Danh sách từ khóa lọc theo 4 nhóm ngành mục tiêu
DOMAIN_KEYWORDS = {
    "IT": [
        "software", "programming", "developer", "data", "database", "cloud", 
        "network", "security", "web", "python", "algorithm", "machine learning",
        "artificial intelligence", "devops", "sql", "linux", "api"
    ],
    "Engineer": [
        "engineering", "mechanical", "electrical", "electronics", "circuit", 
        "cad", "autocad", "solidworks", "robotics", "automation", "plc", 
        "embedded", "firmware", "sensor", "hardware", "mechatronics", "manufacturing"
    ],
    "Economics": [
        "accounting", "finance", "financial", "audit", "marketing", "sales", 
        "management", "business", "tax", "banking", "investment", "budget", 
        "market research", "customer service", "public relations", "human resources"
    ],
    "Healthcare": [
        "medical", "health", "clinical", "nursing", "doctor", "pharmacy", 
        "pharmaceutical", "hospital", "patient", "therapy", "diagnosis", 
        "biomedical", "laboratory", "medicine", "healthcare"
    ]
}

def classify_skill_domain(skill_name: str, alt_labels: str, description: str) -> list:
    """Xác định kỹ năng thuộc nhóm Domain nào."""
    full_text = f"{skill_name} {alt_labels} {description}".lower()
    matched_domains = []
    
    for domain, kws in DOMAIN_KEYWORDS.items():
        pattern = r'\b(' + '|'.join(re.escape(kw) for kw in kws) + r')\b'
        if re.search(pattern, full_text):
            matched_domains.append(domain)
            
    return matched_domains

def main():
    print("=" * 80)
    print("XÂY DỰNG ESCO KNOWLEDGE BASE SUBSET CHO 4 DOMAINS")
    print("=" * 80)

    if not ESCO_SKILLS_CSV.exists():
        print(f"❌ Không tìm thấy file: {ESCO_SKILLS_CSV}")
        print("Vui lòng tải file skills_en.csv và bỏ vào thư mục Data/Raw/ESCO/")
        return

    print(f"⏳ 1. Đang nạp {ESCO_SKILLS_CSV.name}...")
    # ESCO CSV chuẩn chứa các cột: conceptUri, preferredLabel, altLabels, description, skillType
    df = pd.read_csv(ESCO_SKILLS_CSV)
    
    print(f"Tổng số kỹ năng gốc trong ESCO: {len(df)}")

    esco_subset = []
    domain_counts = {"IT": 0, "Engineer": 0, "Economics": 0, "Healthcare": 0}

    print("⏳ 2. Đang lọc và phân loại kỹ năng...")
    for _, row in df.iterrows():
        name = str(row.get('preferredLabel', '')).strip()
        alt = str(row.get('altLabels', '')) if pd.notna(row.get('altLabels')) else ""
        desc = str(row.get('description', '')) if pd.notna(row.get('description')) else ""
        uri = str(row.get('conceptUri', ''))
        
        # Tách các từ đồng nghĩa (altLabels phân tách bằng dấu \n trong ESCO)
        alt_list = [a.strip() for a in alt.split('\n') if a.strip()]

        matched_domains = classify_skill_domain(name, alt, desc)
        
        if matched_domains:
            for d in matched_domains:
                domain_counts[d] += 1
                
            esco_subset.append({
                "uri": uri,
                "skill_name": name,
                "alt_labels": alt_list,
                "domains": matched_domains,
                "description": desc[:300]  # Lưu đoạn mô tả ngắn
            })

    # Lưu file kết quả
    OUTPUT_SUBSET_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUBSET_JSON, "w", encoding="utf-8") as f:
        json.dump(esco_subset, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ HOÀN TẤT TẠO ESCO SUBSET!")
    print(f"Tổng số kỹ năng trích xuất: {len(esco_subset)}")
    print("Phân bổ số lượng kỹ năng theo từng Domain:")
    for d, count in domain_counts.items():
        print(f"   - {d:<12}: {count} skills")
    print(f"File lưu tại: {OUTPUT_SUBSET_JSON}")
    print("=" * 80)

if __name__ == "__main__":
    main()