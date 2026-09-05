from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import docling
import torch
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

from document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass
from ocr_fallback import OCRFallback
from text_cleaner import fix_glued_text

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
_DEFAULT_PARSER: Optional[DocumentParser] = None


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            num_threads=max(os.cpu_count() or 4, 1),
            device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.AUTO,
        ),
        do_ocr=True,
        do_table_structure=True,
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
        self.ocr = OCRFallback(use_gpu=torch.cuda.is_available())

    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return "", {
                "filename": path.name,
                "final_status": "UNSUPPORTED_FORMAT",
                "error": f"Unsupported extension: {path.suffix}",
            }

        record: Dict[str, Any] = {
            "filename": path.name,
            "docling_score": None,
            "ocr_triggered": False,
            "ocr_score": None,
            "final_status": None,
            "error": None,
        }

        # 1. Thử phân tích bằng Docling
        docling_score = None
        cleaned_text = ""
        try:
            result = self.converter.convert(str(path))
            if result and result.document:
                raw_text = result.document.export_to_markdown().strip()
                cleaned_text = fix_glued_text(raw_text)
                docling_score, _ = evaluate_quality(cleaned_text)
                record["docling_score"] = round(docling_score, 3)
                if quality_is_pass(docling_score):
                    record["final_status"] = "ACCEPTED_BY_DOCLING"
                    return cleaned_text, record
        except Exception as exc:
            record["docling_error"] = str(exc)

        # 2. Chuyển sang OCR Fallback nếu Docling lỗi hoặc điểm < 0.65
        try:
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

            return ocr_text, record

        except Exception as exc:
            record["error"] = str(exc)
            record["final_status"] = "FAILED"
            return cleaned_text, record


def get_document_parser() -> DocumentParser:
    global _DEFAULT_PARSER
    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = DocumentParser()
    return _DEFAULT_PARSER