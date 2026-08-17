import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

INPUT_JSON = project_root / "Data" / "Processed" / "cleaned_jds.json"
OUTPUT_FILTERED_JSON = project_root / "Data" / "Processed" / "filtered_jds.json"
OUTPUT_EMBEDDINGS_NPY = project_root / "Data" / "Processed" / "jd_embeddings.npy"

SAMPLES_PER_DOMAIN = 150

def build_jd_text_for_embedding(jd: dict) -> str:
    title = jd.get("job_title", "")
    domain = jd.get("domain", "")
    skills = ", ".join(jd.get("skills", []))
    description = jd.get("description", "")
    desc_preview = description[:1500] if len(description) > 1500 else description
    return f"Job Title: {title}. Domain: {domain}. Required Skills: {skills}. Description: {desc_preview}"

def main():
    print("=" * 80)
    print("LỌC MẪU & VECTOR HÓA KHO JD (PRE-EMBEDDING)")
    print("=" * 80)

    if not INPUT_JSON.exists():
        print(f"❌ Không tìm thấy file: {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_jds = json.load(f)

    domain_buckets = {"IT": [], "Engineering": [], "Economics": []}
    for jd in all_jds:
        domain = jd.get("domain")
        desc = jd.get("description", "")
        if domain in domain_buckets and len(desc.strip()) >= 100:
            domain_buckets[domain].append(jd)

    sampled_jds = []
    print("\n📊 Phân bổ số lượng JD sau khi lấy mẫu:")
    for domain, jds_list in domain_buckets.items():
        selected = jds_list[:SAMPLES_PER_DOMAIN]
        sampled_jds.extend(selected)
        print(f"   - {domain:<12}: {len(selected)} JDs (Gốc: {len(jds_list)})")

    print(f"\nTổng số JD giữ lại: {len(sampled_jds)}")

    print("\n⏳ Đang tải mô hình Sentence Transformer ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("⏳ Đang mã hóa toàn bộ JD sang Vector...")
    jd_texts = [build_jd_text_for_embedding(jd) for jd in sampled_jds]
    embeddings = model.encode(jd_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    OUTPUT_FILTERED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILTERED_JSON, "w", encoding="utf-8") as f:
        json.dump(sampled_jds, f, ensure_ascii=False, indent=2)

    np.save(OUTPUT_EMBEDDINGS_NPY, embeddings)

    print("=" * 80)
    print("✅ HOÀN TẤT PRE-EMBEDDING!")
    print(f"1. Metadata : {OUTPUT_FILTERED_JSON}")
    print(f"2. Vectors  : {OUTPUT_EMBEDDINGS_NPY} (Shape: {embeddings.shape})")
    print("=" * 80)

if __name__ == "__main__":
    main()