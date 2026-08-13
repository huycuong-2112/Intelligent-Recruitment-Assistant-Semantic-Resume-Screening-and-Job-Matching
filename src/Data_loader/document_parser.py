from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from traceback import print_exception
from typing import Any, Dict, List, Tuple

import docling
import torch
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

try:
    from .document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass
    from .ocr_fallback import OCRFallback
    from .image_preprocessor import enhance_image, cleanup_temp
except Exception:
    from document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass
    from ocr_fallback import OCRFallback
    from image_preprocessor import enhance_image, cleanup_temp

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
DEFAULT_RESUME_DIR = Path("Data") / "Raw" / "CrawlResume"
REPORT_PATH = Path("Data") / "Processed" / "document_extraction_report.json"

_DEFAULT_PARSER: DocumentParser | None = None


def _select_accelerator_device() -> AcceleratorDevice:
    return AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.AUTO


def _build_converter() -> DocumentConverter:
    # Configure pipeline with layout analysis and OCR enabled
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            num_threads=max(os.cpu_count() or 4, 1),
            device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.AUTO,
        ),
        do_ocr=True,
        do_table_structure=True,  # Enables structure recognition for tables/grids
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        },
    )


class DocumentParser:
    def __init__(self) -> None:
        self.converter = _build_converter()
        self.chunker = HybridChunker()
        self.ocr = OCRFallback(use_gpu=torch.cuda.is_available())

    def parse_to_chunks(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            """
            Parses document using Docling layout model and breaks it down into 
            hierarchical chunks with preserved heading anchors.
            """
            path = Path(file_path)
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")

            record = {
                "filename": path.name,
                "docling_score": None,
                "ocr_triggered": False,
                "ocr_score": None,
                "final_status": None,
                "error": None,
            }

            try:
                # 1. Primary conversion with Layout Analysis
                result = self.converter.convert(str(path))
                doc = result.document

                if not doc:
                    raise RuntimeError("Docling failed to produce a valid document object.")

                # Extract raw Markdown for Quality Gate Check
                raw_text = doc.export_to_markdown().strip()
                cleaned_text = raw_text

                docling_score, _ = evaluate_quality(cleaned_text)
                record["docling_score"] = round(docling_score, 3)

                chunks_output: List[Dict[str, Any]] = []

                # 2. If passes Quality Gate, perform Layout Chunking
                if quality_is_pass(docling_score):
                    record["final_status"] = "ACCEPTED_BY_DOCLING"
                    
                    # Run layout-aware chunking
                    raw_chunks = list(self.chunker.chunk(doc))
                    
                    for idx, chunk in enumerate(raw_chunks):
                        # Extract associated heading path for context retention
                        headings = getattr(chunk.meta, "headings", []) if hasattr(chunk, "meta") else []
                        
                        chunks_output.append({
                            "chunk_id": idx,
                            "headings": headings,  # e.g., ["KINH NGHIỆM LÀM VIỆC", "IT Support Specialist"]
                            "text": fix_glued_text(chunk.text),
                        })
                    return chunks_output, record

                # 3. Fallback to OCR if Layout Parsing Quality Gate Fails
                record["ocr_triggered"] = True
                raw_ocr = self.ocr.extract(str(path))
                ocr_text = fix_glued_text(raw_ocr)
                ocr_score, _ = evaluate_quality(ocr_text)
                record["ocr_score"] = round(ocr_score, 3)

                if quality_is_pass(ocr_score) or (
                    ocr_score is not None and docling_score is not None and ocr_score > docling_score
                ):
                    record["final_status"] = "RECOVERED_BY_OCR"
                else:
                    record["final_status"] = "LOW_QUALITY"

                # Fallback chunk output (single section dump)
                chunks_output.append({
                    "chunk_id": 0,
                    "headings": ["OCR_FALLBACK_TEXT"],
                    "text": ocr_text,
                })

                return chunks_output, record

            except Exception as exc:
                record["error"] = str(exc)
                record["final_status"] = "FAILED"
                return [], record


def get_document_parser() -> DocumentParser:
    global _DEFAULT_PARSER
    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = DocumentParser()
    return _DEFAULT_PARSER


def parse_document(file_path: str) -> str:
    text, _ = get_document_parser().parse(file_path)
    return text


def _process_file(parser: DocumentParser, file_path: Path, report_list: List[Dict[str, Any]]) -> bool:
    print(f"\nProcessing: {file_path.name}")
    final_text, record = parser.parse(str(file_path))
    report_list.append(record)

    if record["final_status"] in ("ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR"):
        print(f"Status: {record['final_status']} | Score: {record.get('ocr_score') or record.get('docling_score')}")
        print("--- Text Sample ---")
        print("\n".join(final_text.splitlines()[:10]))
        return True
    
    print(f"Failed or Low Quality: {record['final_status']} (Error: {record.get('error')})")
    return False