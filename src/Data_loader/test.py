from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from shutil import which
from pathlib import Path
from traceback import print_exception
from typing import Any, Dict, List, Tuple

import docling
import torch
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

try:
    from .document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass, QUALITY_THRESHOLD
    from .ocr_fallback import OCRFallback
    from .image_preprocessor import enhance_image, cleanup_temp
except Exception:
    from document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass, QUALITY_THRESHOLD  # type: ignore
    from ocr_fallback import OCRFallback  # type: ignore
    from image_preprocessor import enhance_image, cleanup_temp  # type: ignore

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
DEFAULT_RESUME_DIR = Path("Data") / "Raw" / "CrawlResume"
_DEFAULT_PARSER: DocumentParser | None = None
REPORT_PATH = Path("Data") / "Processed" / "document_extraction_report.json"


def _select_accelerator_device() -> AcceleratorDevice:
    if torch.cuda.is_available():
        return AcceleratorDevice.CUDA
    return AcceleratorDevice.AUTO


def _create_pipeline_options() -> PdfPipelineOptions:
    return PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            num_threads=max(os.cpu_count() or 4, 1),
            device=_select_accelerator_device(),
        )
    )


def _build_converter() -> DocumentConverter:
    pdf_pipeline_options = _create_pipeline_options()
    image_pipeline_options = _create_pipeline_options()

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_pipeline_options),
        },
    )


class DocumentParser:
    """Parse resume documents using Docling with quality gating and OCR fallback."""

    def __init__(self) -> None:
        self.converter = _build_converter()
        self.ocr = OCRFallback(use_gpu=torch.cuda.is_available())

    def _convert_with_docling(self, path: str):
        return self.converter.convert(path)

    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Thực hiện toàn bộ quy trình Ingestion:
        Docling -> Quality Gate -> OCR Fallback (nếu đứt gate) -> Trả về (final_text, audit_report)
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        record: Dict[str, Any] = {
            "filename": path.name,
            "docling_score": None,
            "ocr_triggered": False,
            "ocr_score": None,
            "final_status": None,
            "error": None,
        }

        try:
            # 1. Chạy Docling
            result = self._convert_with_docling(str(path))
            document = result.document
            if document is None:
                raise RuntimeError(f"Docling did not return a document for '{path}'")
            
            docling_text = document.export_to_markdown().strip()
            
            # 2. Chấm điểm Quality Gate
            docling_score, metrics = evaluate_quality(docling_text)
            record["docling_score"] = round(docling_score, 3)
            final_text = docling_text

            # 3. Kiểm tra Quality Gate -> Nếu Fail thì chạy OCR Fallback
            if not quality_is_pass(docling_score):
                record["ocr_triggered"] = True
                print(f"[{path.name}] Quality gate failed ({docling_score:.3f}) — Running OCR fallback...")
                
                tmp_path = None
                try:
                    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        tmp_path = enhance_image(str(path), scale=1.0)
                        ocr_text = self.ocr.extract(tmp_path)
                    else:
                        ocr_text = self.ocr.extract(str(path))

                    ocr_score, _ = evaluate_quality(ocr_text)
                    record["ocr_score"] = round(ocr_score, 3)

                    if quality_is_pass(ocr_score) or (ocr_score and ocr_score > docling_score):
                        final_text = ocr_text
                        record["final_status"] = "RECOVERED_BY_OCR"
                    else:
                        record["final_status"] = "LOW_QUALITY"

                except Exception as exc_ocr:
                    record["error"] = f"OCR failed: {exc_ocr}"
                    record["final_status"] = "OCR_FAILED"
                finally:
                    if tmp_path:
                        cleanup_temp(tmp_path)
            else:
                record["final_status"] = "ACCEPTED_BY_DOCLING"

            return final_text, record

        except Exception as exc:
            record["error"] = str(exc)
            record["final_status"] = "FAILED"
            return "", record


def get_document_parser() -> DocumentParser:
    global _DEFAULT_PARSER
    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = DocumentParser()
    return _DEFAULT_PARSER


def parse_document(file_path: str) -> Tuple[str, Dict[str, Any]]:
    return get_document_parser().parse(file_path)


def _process_file(parser: DocumentParser, file_path: Path, report_list: List[Dict[str, Any]]) -> bool:
    print("\n" + "=" * 80)
    print(f"Processing: {file_path.name}")
    print("=" * 80)
    
    # Đã giải nén Tuple an toàn
    final_text, record = parser.parse(str(file_path))
    report_list.append(record)

    if record.get("final_status") in ("ACCEPTED_BY_DOCLING", "RECOVERED_BY_OCR"):
        print(f"Status: {record['final_status']} | Text length: {len(final_text)}")
        print("\n--- Extracted text excerpt ---")
        print("\n".join(final_text.splitlines()[:15]))
        return True
    else:
        print(f"Status: {record['final_status']} | Error: {record.get('error')}")
        return False