import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Tự động tìm thư mục gốc Project chứa folder "Data"
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

INPUT_JSON = project_root / "Data" / "Processed" / "cleaned_jds.json"
OUTPUT_FILTERED_JSON = project_root / "Data" / "Processed" / "filtered_jds.json"
OUTPUT_EMBEDDINGS_NPY = project_root / "Data" / "Processed" / "jd_embeddings.npy"

# Số lượng JD tối đa muốn lấy cho mỗi Domain (để tránh nặng RAM)
SAMPLES_PER_DOMAIN = 150

def build_jd_text_for_embedding(jd: dict) -> str:
    """Tạo đoạn văn bản tổng hợp từ JD để mô hình hiểu toàn diện ngữ cảnh."""
    title = jd.get("job_title", "")
    domain = jd.get("domain", "")
    skills = ", ".join(jd.get("skills", []))
    description = jd.get("description", "")
    
    # Cắt ngắn description nếu quá dài để tối ưu tốc độ embedding (lấy tối đa 1500 ký tự đầu)
    desc_preview = description[:1500] if len(description) > 1500 else description
    
    return f"Job Title: {title}. Domain: {domain}. Required Skills: {skills}. Description: {desc_preview}"

def main():
    print("=" * 80)
    print("BƯỚC CUỐI XỬ LÝ JD: LỌC MẪU & VECTOR HÓA (PRE-EMBEDDING)")
    print("=" * 80)

    if not INPUT_JSON.exists():
        print(f"❌ Không tìm thấy file: {INPUT_JSON}")
        print("Hãy chắc chắn bạn đã chạy file process_jds.py trước!")
        return

    print(f"⏳ 1. Đang nạp dữ liệu từ: {INPUT_JSON.name}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_jds = json.load(all_jds_file := open(INPUT_JSON, "r", encoding="utf-8"))

    # Gom nhóm theo Domain và lọc chất lượng
    domain_buckets = {"IT": [], "Engineer": [], "Economics": [], "Healthcare": []}
    
    for jd in all_jds:
        domain = jd.get("domain")
        desc = jd.get("description", "")
        # Lọc cơ bản: bỏ JD có mô tả quá ngắn dưới 100 ký tự
        if domain in domain_buckets and len(desc.strip()) >= 100:
            domain_buckets[domain].append(jd)

    # Lấy mẫu đại diện
    sampled_jds = []
    print("\n📊 Phân bổ số lượng JD sau khi lấy mẫu:")
    for domain, jds_list in domain_buckets.items():
        # Lấy tối đa SAMPLES_PER_DOMAIN bản ghi
        selected = jds_list[:SAMPLES_PER_DOMAIN]
        sampled_jds.extend(selected)
        print(f"   - {domain:<12}: {len(selected)} JDs (Gốc: {len(jds_list)})")

    print(f"\nTổng số JD được giữ lại: {len(sampled_jds)}")

    # Vector hóa bằng SentenceTransformer
    print("\n⏳ 2. Đang tải mô hình Sentence Transformer ('all-MiniLM-L6-v2')...")
    print("*(Lần chạy đầu tiên sẽ mất khoảng 10-20 giây để tải mô hình ~90MB về máy)*")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("⏳ 3. Đang mã hóa toàn bộ JD sang Vector...")
    jd_texts = [build_jd_text_for_embedding(jd) for jd in sampled_jds]
    
    # Tiến hành encode thành ma trận numpy (batch_size=32 để chạy mượt)
    embeddings = model.encode(jd_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    # Lưu dữ liệu
    print("\n💾 4. Đang lưu trữ kết quả...")
    OUTPUT_FILTERED_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu file JSON metadata
    with open(OUTPUT_FILTERED_JSON, "w", encoding="utf-8") as f:
        json.dump(sampled_jds, f, ensure_ascii=False, indent=2)

    # Lưu file Vector npy
    np.save(OUTPUT_EMBEDDINGS_NPY, embeddings)

    print("=" * 80)
    print("✅ HOÀN TẤT TOÀN BỘ DỮ LIỆU JD!")
    print(f"1. File Metadata : {OUTPUT_FILTERED_JSON} (Dùng để hiển thị thông tin JD)")
    print(f"2. File Vectors  : {OUTPUT_EMBEDDINGS_NPY} (Shape: {embeddings.shape})")
    print("=" * 80)

if __name__ == "__main__":
    main()