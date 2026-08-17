from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Tự động tìm thư mục gốc Project chứa folder "Data"
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "Data").exists():
        break
    project_root = project_root.parent

# Thêm đường dẫn project_root vào sys.path để tránh lỗi ImportError
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from document_parser import get_document_parser, SUPPORTED_EXTENSIONS
except ImportError:
    try:
        from src.Data_loader.document_parser import get_document_parser, SUPPORTED_EXTENSIONS
    except ImportError:
        from .document_parser import get_document_parser, SUPPORTED_EXTENSIONS  # type: ignore

# Trỏ trực tiếp vào thư mục Data/Raw/Resumes (chứa 3 folder IT, Engineer, Economics)
DEFAULT_INPUT_DIR = project_root / "Data" / "Raw" / "Resumes"
DEFAULT_OUTPUT_DIR = project_root / "Data" / "Processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from CV documents and generate audit report + cleaned dataset."
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--file", type=str, default=None)
    return parser.parse_args()


def collect_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "document_extraction_report.json"
    cleaned_text_path = output_dir / "cleaned_text.json"

    print("=" * 80)
    print("DOCUMENT EXTRACTION PIPELINE")
    print(f"Project Root    : {project_root}")
    print(f"Input Directory : {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    doc_parser = get_document_parser()

    if args.file:
        single_file = Path(args.file)
        if not single_file.is_absolute():
            single_file = Path.cwd() / single_file
        files = [single_file] if single_file.exists() else []
    else:
        files = collect_files(input_dir)

    if not files:
        print(f"❌ Error: No supported resume files found in {input_dir}")
        return 1

    print(f"📂 Found {len(files)} file(s) to process.\n")

    report_items: List[Dict[str, Any]] = []
    cleaned_texts: List[Dict[str, Any]] = []

    success_count = 0
    failure_count = 0

    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing: {file_path.name}...")

        chunks, record = doc_parser.parse_to_chunks(str(file_path))
        report_items.append(record)

        status = record.get("final_status")

        if status in ("ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR", "LOW_QUALITY"):
            success_count += 1
            full_text = "\n\n".join(chunk["text"] for chunk in chunks)
            
            try:
                rel_path = str(file_path.relative_to(project_root))
            except ValueError:
                rel_path = str(file_path)

            cleaned_texts.append({
                "filename": file_path.name,
                "relative_path": rel_path,
                "status": status,
                "total_chunks": len(chunks),
                "text_length": len(full_text),
                "chunks": chunks,
                "content": full_text,
            })
            print(f"   └─ Status: {status} | Chunks: {len(chunks)} | Chars: {len(full_text)}")
        else:
            failure_count += 1
            print(f"   └─ Status: {status} | Error: {record.get('error')}")

    report_summary = {
        "total_files": len(files),
        "successful_extractions": success_count,
        "failed_extractions": failure_count,
        "status_breakdown": {
            status: sum(1 for item in report_items if item.get("final_status") == status)
            for status in set(item.get("final_status") for item in report_items)
        },
    }

    report_output = {
        "summary": report_summary,
        "details": report_items,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_output, f, ensure_ascii=False, indent=2)

    with open(cleaned_text_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_texts, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print(f"Total Processed : {len(files)}")
    print(f"Success         : {success_count}")
    print(f"Failed          : {failure_count}")
    print(f"Report File     : {report_path}")
    print(f"Cleaned Text    : {cleaned_text_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())