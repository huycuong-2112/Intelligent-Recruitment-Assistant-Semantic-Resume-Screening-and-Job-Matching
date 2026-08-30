import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Data_loader.offline.jd_offline_parser import OfflineJDExtractor

def main():
    input_file = "Data/Processed/cleaned_jds.json"
    output_file = "Data/Processed/offline_jd_demo_output.json"
    
    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        input_file = sys.argv[idx + 1]
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output_file = sys.argv[idx + 1]

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        jds = json.load(f)

    print(f"Loaded {len(jds)} JDs. Starting offline parsing...\n")

    results = []
    
    for jd_raw in jds:
        jd_id = jd_raw.get("id", "unknown_id")
        filename = jd_raw.get("filename", "")
        status = jd_raw.get("status", "")
        text = jd_raw.get("content", "")

        fallback_title = filename.replace(".pdf", "").replace(".docx", "")
        
        parsed_jd = OfflineJDExtractor.parse(text, fallback_title=fallback_title)
        
        # Serialize and attach metadata
        jd_dict = parsed_jd.dict()
        jd_dict["id"] = jd_id
        jd_dict["filename"] = filename
        jd_dict["source_status"] = status
        jd_dict["extraction_method"] = "offline_hybrid"

        results.append(jd_dict)

        print(f"[offline] {jd_id} -> {jd_dict['job_title']}")
        print(f"  exp: {jd_dict['min_experience_years']}")
        print(f"  degree: {jd_dict['required_degree']}")
        print(f"  fields: {jd_dict['preferred_fields']}")
        print(f"  required_skills: {jd_dict['required_skills']}")
        print(f"  preferred_skills: {jd_dict['preferred_skills']}")
        print(f"  responsibilities: {len(jd_dict['responsibilities'])}")
        print(f"  key_deliverables: {len(jd_dict['key_deliverables'])}")
        print(f"  certifications: {jd_dict['required_certifications']}")
        print("-" * 50)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Wrote {len(results)} parsed JDs to {output_file}")

if __name__ == "__main__":
    main()
