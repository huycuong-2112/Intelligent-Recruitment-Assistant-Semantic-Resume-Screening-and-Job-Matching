import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Suppress verbose warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU execution
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Docling Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions


# =====================================================================
# STEP 1: Pydantic Data Schema Definition
# =====================================================================
class ResumeSchema(BaseModel):
    id: str = Field(description="Unique identifier for the CV")
    filename: str = Field(description="Original filename of the CV")
    raw_markdown: str = Field(default="", description="Full Markdown extracted by Docling")
    
    skills: List[str] = Field(
        default_factory=list, 
        description="Extracted technical and soft skills"
    )
    experience_years: float = Field(
        default=0.0, ge=0.0, 
        description="Total estimated years of professional experience"
    )
    education_degree: Literal["PhD", "Master", "Bachelor", "Diploma", "None"] = Field(
        default="Bachelor", 
        description="Highest degree level attained"
    )
    education_field: str = Field(
        default="General", 
        description="Major or field of study"
    )
    job_titles: List[str] = Field(
        default_factory=list, 
        description="List of past job positions held"
    )


# =====================================================================
# STEP 2: Docling + EasyOCR Document Extractor
# =====================================================================
class DoclingEasyOCRExtractor:
    """
    Extracts layout-aware Markdown from digital or scanned PDFs/images
    using Docling wrapped around EasyOCR.
    """
    def __init__(self):
        logging.info("Initializing Docling with EasyOCR engine...")
        
        # Configure PDF pipeline to perform EasyOCR when necessary
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)  # Hybrid mode
        
        # Instantiate converter with custom PDF format options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract_markdown(self, file_path: Path) -> str:
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        logging.info(f"Converting document via Docling + EasyOCR: {file_path.name}")
        result = self.converter.convert(str(file_path))
        markdown_text = result.document.export_to_markdown()
        return markdown_text.strip()


# =====================================================================
# STEP 3: MiniLM Semantic Header & Offline Fallback Parser
# =====================================================================
class MiniLMFallbackParser:
    """
    Parses Markdown sections using MiniLM vector similarity to classify
    custom section headers into standard target categories offline.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logging.info(f"Loading MiniLM model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        # Target canonical concepts to match section headers against
        self.target_phrases = {
            "skills": ["skills", "technical skills", "competencies", "tools", "programming languages", "technologies"],
            "experience": ["work experience", "professional experience", "employment history", "career", "work history"],
            "education": ["education", "academic background", "qualifications", "degrees", "university"]
        }
        
        # Pre-compute target vectors for speed ($O(1)$ runtime lookup)
        self.target_embeddings = {
            category: self.embedder.encode(phrases, convert_to_tensor=True)
            for category, phrases in self.target_phrases.items()
        }

    def _split_by_markdown_headers(self, md_text: str) -> Dict[str, str]:
        """Splits Markdown text into {header: section_content} blocks."""
        sections = {}
        header_pattern = r'^(#{1,3})\s+(.+)$'
        lines = md_text.split('\n')
        
        current_header = "General Header"
        current_body = []
        
        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                if current_body:
                    sections[current_header] = "\n".join(current_body).strip()
                    current_body = []
                current_header = match.group(2).strip()
            else:
                current_body.append(line)
                
        if current_body:
            sections[current_header] = "\n".join(current_body).strip()
            
        return sections

    def _classify_header(self, header_title: str, sim_threshold: float = 0.40) -> str:
        """Computes Cosine Similarity between a header and target concepts."""
        header_vector = self.embedder.encode(header_title, convert_to_tensor=True)
        best_category = "unknown"
        best_score = -1.0
        
        for category, target_vectors in self.target_embeddings.items():
            sims = self.embedder.similarity(header_vector, target_vectors)
            max_sim = float(sims.max())
            if max_sim > best_score and max_sim >= sim_threshold:
                best_score = max_sim
                best_category = category
                
        return best_category

    def parse(self, md_text: str, cv_id: str, filename: str) -> ResumeSchema:
        """Transforms raw Markdown into a validated Pydantic ResumeSchema."""
        sections = self._split_by_markdown_headers(md_text)
        
        raw_skills = []
        experience_text = ""
        education_text = ""
        
        # Map sections via MiniLM cosine similarity
        for header, content in sections.items():
            category = self._classify_header(header)
            
            if category == "skills":
                # Clean list items (- Python, * Docker, or comma-separated)
                lines = [l.strip("-*• ").strip() for l in content.split("\n") if l.strip()]
                for line in lines:
                    raw_skills.extend([s.strip() for s in line.split(",") if s.strip()])
                    
            elif category == "experience":
                experience_text += content + "\n"
                
            elif category == "education":
                education_text += content + "\n"

        # Heuristic extraction for experience years
        exp_matches = re.findall(r'(\d+)\+?\s*(?:years?|năm)', experience_text, re.IGNORECASE)
        years = float(max([int(y) for y in exp_matches if int(y) <= 30], default=0.0))

        # Heuristic degree level classification
        degree = "Bachelor"
        edu_lower = education_text.lower()
        if any(k in edu_lower for k in ["phd", "ph.d", "doctorate", "tiến sĩ"]):
            degree = "PhD"
        elif any(k in edu_lower for k in ["master", "m.s", "m.sc", "m.a", "thạc sĩ"]):
            degree = "Master"
        elif any(k in edu_lower for k in ["diploma", "associate", "cao đẳng"]):
            degree = "Diploma"

        # Construct and validate final Pydantic object
        return ResumeSchema(
            id=cv_id,
            filename=filename,
            raw_markdown=md_text,
            skills=list(set(raw_skills)),
            experience_years=years,
            education_degree=degree,
            education_field="General",
            job_titles=[]
        )


# =====================================================================
# STEP 4: Main End-to-End Execution Pipeline
# =====================================================================
class CVProcessingPipeline:
    def __init__(self):
        self.docling_extractor = DoclingEasyOCRExtractor()
        self.minilm_parser = MiniLMFallbackParser()

    def process_file(self, file_path: str | Path, cv_id: str = "cv_001") -> ResumeSchema:
        path = Path(file_path)
        
        # 1. Extract Markdown via Docling + EasyOCR
        markdown_content = self.docling_extractor.extract_markdown(path)
        
        # 2. Classify headers and build structured output via MiniLM + Pydantic
        resume_schema = self.minilm_parser.parse(
            md_text=markdown_content, 
            cv_id=cv_id, 
            filename=path.name
        )
        
        return resume_schema


# =====================================================================
# STEP 5: Verification & Testing
# =====================================================================
if __name__ == "__main__":
    # Initialize full pipeline once
    pipeline = CVProcessingPipeline()
    
    # Path to sample PDF or scanned image CV
    sample_file = Path("data/raw/sample_resume.pdf")

    if sample_file.exists():
        print("\n" + "=" * 60)
        print("RUNNING PIPELINE")
        print("=" * 60)
        
        # Execute Pipeline
        result: ResumeSchema = pipeline.process_file(sample_file, cv_id="cv_101")
        
        # Print validated Pydantic model details
        print("\n=== PIPELINE SUCCESSFUL ===")
        print(f"ID: {result.id}")
        print(f"Filename: {result.filename}")
        print(f"Extracted Skills ({len(result.skills)}): {result.skills[:5]}")
        print(f"Experience Years: {result.experience_years}")
        print(f"Education Degree: {result.education_degree}")
        
        # Dump full validated JSON
        print("\n=== VALIDATED PYDANTIC JSON OUTPUT ===")
        print(result.model_dump_json(indent=2))
    else:
        print(f"\n[Note] Place a test CV at '{sample_file}' to test this script directly.")