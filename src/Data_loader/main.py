from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

current_file = Path(__file__).resolve()
current_dir = current_file.parent
project_root = current_dir

while project_root != project_root.parent:
    if (project_root / "Data").exists() or (project_root / "src").exists():
        break
    project_root = project_root.parent

for path_entry in [str(current_dir), str(project_root)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

from document_parser import get_document_parser, SUPPORTED_EXTENSIONS

DEFAULT_RESUME_DIR = project_root / "Data" / "Raw" / "Resumes"
DEFAULT_JD_DIR = project_root / "Data" / "Raw" / "JD"
DEFAULT_OUTPUT_DIR = project_root / "Data" / "Processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal Stage 1: Document Text Ingestion & Cleaning")
    parser.add_argument("--type", choices=["resumes", "jds", "custom"], default="resumes", help="Target entity type")
    parser.add_argument("--input-dir", type=str, default=None)
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
    if args.type == "resumes":
        input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_RESUME_DIR
        out_filename = "cleaned_resumes.json"
        prefix = "cv"
    elif args.type == "jds":
        input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_JD_DIR
        out_filename = "cleaned_jds.json"
        prefix = "jd"
    else:
        input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_RESUME_DIR
        out_filename = "cleaned_custom.json"
        prefix = "doc"
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{args.type}_extraction_report.json"
    cleaned_text_path = output_dir / out_filename

    print("=" * 80)
    print(f"STAGE 1 EXTRACTION: {args.type.upper()}")
    print(f"Input Directory : {input_dir}")
    print(f"Output File     : {cleaned_text_path}")
    print("=" * 80)

    doc_parser = get_document_parser()
    files = [Path(args.file)] if args.file else collect_files(input_dir)

    if not files:
        print(f"⚠️ No documents found in {input_dir}")
        return 0

    print(f"📂 Found {len(files)} file(s) to process.\n")

    report_items: List[Dict[str, Any]] = []
    cleaned_texts: List[Dict[str, Any]] = []

    success_count = 0
    failure_count = 0

    for idx, file_path in enumerate(files, start=1):
        doc_id = f"{prefix}_{idx:03d}"
        print(f"[{idx}/{len(files)}] Ingesting: {file_path.name}...")

        content, record = doc_parser.parse(str(file_path))
        status = record.get("final_status")
        report_items.append(record)

        if status in ("ACCEPTED_DIRECT_TEXT","ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR", "LOW_QUALITY"):
            success_count += 1
            try:
                rel_path = str(file_path.relative_to(project_root))
            except ValueError:
                rel_path = str(file_path)

            cleaned_texts.append({
                "id": doc_id,
                "filename": file_path.name,
                "relative_path": rel_path,
                "status": status,
                "text_length": len(content),
                "content": content,
            })
            print(f"   └─ Status: {status} | Characters Extracted: {len(content):,}")
        else:
            failure_count += 1
            print(f"   └─ Status: {status} | Error: {record.get('error')}")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(files), "success": success_count, "details": report_items}, f, indent=2, ensure_ascii=False)

    with open(cleaned_text_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_texts, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"INGESTION COMPLETE: {success_count}/{len(files)} successful -> {cleaned_text_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())