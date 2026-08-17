from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
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
    from .text_cleaner import fix_glued_text
except Exception:
    from document_quality import evaluate as evaluate_quality, is_pass as quality_is_pass
    from ocr_fallback import OCRFallback
    from image_preprocessor import enhance_image, cleanup_temp
    from text_cleaner import fix_glued_text

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
_DEFAULT_PARSER: DocumentParser | None = None


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
        self.chunker = HybridChunker()
        self.ocr = OCRFallback(use_gpu=torch.cuda.is_available())

    def parse_to_chunks(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return [], {
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

        docling_succeeded = False
        doc = None
        cleaned_text = ""
        docling_score = None

        # 1. Thử phân tích tài liệu bằng Docling
        try:
            result = self.converter.convert(str(path))
            doc = result.document
            if doc:
                raw_text = doc.export_to_markdown().strip()
                cleaned_text = fix_glued_text(raw_text)
                docling_score, _ = evaluate_quality(cleaned_text)
                record["docling_score"] = round(docling_score, 3)
                docling_succeeded = quality_is_pass(docling_score)
        except Exception as exc:
            # Ghi nhận lỗi nhưng không return để tiếp tục nhảy sang OCR Fallback
            record["docling_error"] = str(exc)

        # 2. Nếu Docling đạt chất lượng, tiến hành Chunking
        if docling_succeeded and doc:
            record["final_status"] = "ACCEPTED_BY_DOCLING"
            chunks_output: List[Dict[str, Any]] = []
            raw_chunks = list(self.chunker.chunk(doc))
            
            for idx, chunk in enumerate(raw_chunks):
                headings = getattr(chunk.meta, "headings", []) if hasattr(chunk, "meta") else []
                chunks_output.append({
                    "chunk_id": idx,
                    "headings": headings,
                    "text": fix_glued_text(chunk.text),
                })
            return chunks_output, record

        # 3. Kích hoạt OCR Fallback nếu Docling thất bại hoặc chất lượng kém
        try:
            record["ocr_triggered"] = True
            
            # Tiền xử lý ảnh nếu là file ảnh để tăng độ chính xác
            target_path = str(path)
            enhanced_path = None
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                try:
                    enhanced_path = enhance_image(target_path)
                    target_path = enhanced_path
                except Exception:
                    pass

            raw_ocr = self.ocr.extract(target_path)
            ocr_text = fix_glued_text(raw_ocr)
            ocr_score, _ = evaluate_quality(ocr_text)
            record["ocr_score"] = round(ocr_score, 3)

            # Dọn dẹp file ảnh tạm nếu có
            if enhanced_path:
                try:
                    cleanup_temp(enhanced_path)
                except Exception:
                    pass

            if quality_is_pass(ocr_score) or (
                ocr_score is not None and docling_score is not None and ocr_score > docling_score
            ):
                record["final_status"] = "RECOVERED_BY_OCR"
            else:
                record["final_status"] = "LOW_QUALITY"

            chunks_output = [{
                "chunk_id": 0,
                "headings": ["OCR_FALLBACK_TEXT"],
                "text": ocr_text,
            }]
            return chunks_output, record

        except Exception as exc:
            record["error"] = str(exc)
            record["final_status"] = "FAILED"
            return [], record

    def parse(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        chunks, record = self.parse_to_chunks(file_path)
        full_text = "\n\n".join(chunk["text"] for chunk in chunks)
        return full_text, record


def get_document_parser() -> DocumentParser:
    global _DEFAULT_PARSER
    if _DEFAULT_PARSER is None:
        _DEFAULT_PARSER = DocumentParser()
    return _DEFAULT_PARSER


def parse_document(file_path: str) -> str:
    text, _ = get_document_parser().parse(file_path)
    return text