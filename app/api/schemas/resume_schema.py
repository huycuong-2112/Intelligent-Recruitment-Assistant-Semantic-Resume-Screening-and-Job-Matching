from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ExtractionMetadata(BaseModel):
    status: str
    text_length: int
    docling_score: Optional[float] = None
    ocr_triggered: bool = False
    ocr_score: Optional[float] = None
    warning: Optional[str] = None


class ResumeParseResponse(BaseModel):
    run_id: str
    document_id: str
    filename: str
    extraction: ExtractionMetadata
    parsed: Dict[str, Any]
    ui_features: List[Dict[str, Any]]

class ConfirmRequest(BaseModel):
    run_id: str
    document_id: str
    confirmed_features: List[Dict[str, Any]]

class ConfirmResponse(BaseModel):
    run_id: str
    document_id: str
    status: str
    override: Dict[str, Any]
    runtime_parsed: Dict[str, Any]
    applied_actions: List[Dict[str, Any]]
    unsupported_actions: List[Dict[str, Any]]
