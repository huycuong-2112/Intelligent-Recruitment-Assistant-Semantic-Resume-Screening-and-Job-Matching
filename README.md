# Intelligent Recruitment Assistant
### Semantic Resume Screening & Job Matching with MDMS and Explainable AI

An end-to-end recruitment-assistance system for parsing CVs and Job Descriptions (JDs), normalizing and representing candidate/job information, ranking candidates with a **Multi-Dimension Matching Score (MDMS)**, and generating evidence-grounded explanations for HR review.

> **Current status:** Local MVP / research prototype.  
> The production runtime path is designed as **Streamlit → FastAPI → NLP/Matching core**.  
> Research experiments are kept separate from runtime artifacts.

---

## 1. Key Features

- Upload and parse **CVs** and **Job Descriptions**
- Supports PDF and image-based documents
- Document extraction with **Docling**, quality checks, and OCR fallback
- Structured CV/JD extraction with **Groq LLM when configured**
- Deterministic offline fallback when Groq is unavailable
- Human **Refine / Confirm** step before downstream matching
- NLP normalization for skills, experience, education, and job requirements
- Local semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Four-dimension candidate matching:
  - Skill
  - Experience
  - Education
  - Semantic similarity
- Weighted **MDMS** aggregation and candidate ranking
- Evidence-grounded **XAI** layer
- Candidate report / explanation generation with deterministic guardrails
- Development evaluation pipeline with ranking and error metrics
- Experimental Skill Matcher V2 research kept isolated from production V1

---

## 2. System Architecture

```mermaid
flowchart TD
    A[CV / JD Upload] --> B[Document Extraction]
    B --> B1[Docling]
    B --> B2[OCR Fallback]
    B1 --> C[Structured Parsing]
    B2 --> C

    C --> C1[Groq LLM - optional]
    C --> C2[Offline Hybrid Fallback]

    C1 --> D[Refine & Confirm]
    C2 --> D

    D --> E[Normalization]
    E --> F[Feature & Chunk Building]
    F --> G[MiniLM Embeddings]

    G --> H1[Skill Matcher]
    G --> H2[Experience Matcher]
    G --> H3[Education Matcher]
    G --> H4[Semantic Matcher]

    H1 --> I[MDMS Aggregation]
    H2 --> I
    H3 --> I
    H4 --> I

    I --> J[Candidate Ranking]
    J --> K[Deterministic XAI]
    K --> L[Candidate Explanation]
    L --> M[Streamlit UI]
```

### Runtime ownership

```text
Streamlit UI
    ↓
FastAPI
    ↓
Runtime services
    ↓
src/ NLP core
    ↓
Persisted runtime artifacts
```

The frontend does not own scoring truth. Matching, evidence, XAI, and explanations are produced by backend/runtime services.

---

## 3. End-to-End Pipeline

### Stage 1 — Document ingestion

Supported inputs include:

- `.pdf`
- `.png`
- `.jpg`
- `.jpeg`

The document parser first attempts structured extraction using **Docling**. If document quality is insufficient, the pipeline can fall back to OCR.

Typical source-quality statuses include:

```text
ACCEPTED_BY_DOCLING
RECOVERED_BY_OCR
LOW_QUALITY
```

### Stage 2 — Structured CV/JD extraction

Extracted text is converted into structured JSON.

When `GROQ_API_KEY` is configured:

```text
text
→ Groq structured extraction
→ extraction_method = groq_llm
```

If Groq is unavailable or fails:

```text
text
→ deterministic/offline parser
→ extraction_method = offline_hybrid
```

### Stage 3 — Refine and Confirm

Users may review extracted information before matching.

Confirmed runtime documents become the authoritative downstream input.

```text
Parsed document
→ Refine
→ Confirm
→ Runtime document
```

### Stage 4 — NLP preparation

Confirmed documents are transformed through:

```text
Normalization
→ Feature building
→ Evidence chunking
→ Profile construction
→ Embedding generation
```

The current local embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized 384-dimensional vectors.

### Stage 5 — Multi-Dimension Matching

The current MDMS architecture evaluates four dimensions:

| Dimension | Description |
|---|---|
| Skill | Match JD skill requirements against candidate skill evidence |
| Experience | Experience years and responsibility-level evidence |
| Education | Degree and field alignment |
| Semantic | Global CV-to-JD semantic alignment |

Current top-level development-selected weights:

```text
Skill      = 0.40
Experience = 0.20
Education  = 0.10
Semantic   = 0.30
```

Conceptually:

```text
MDMS =
    0.40 × S_skill
  + 0.20 × S_experience
  + 0.10 × S_education
  + 0.30 × S_semantic
```

The canonical backend score is on a **0–1 scale**.

For UI display:

```text
score_0_3 = score_0_1 × 3
```

> Missing/unknown components are handled explicitly by runtime semantics.  
> An evaluated `0.0` is not the same as an unavailable (`None`) score.

### Stage 6 — Ranking

Candidates are ranked from persisted MDMS results.

Ranking does **not** recompute matching scores.

### Stage 7 — Explainability

The XAI pipeline converts deterministic matcher evidence into an evidence registry and candidate-level explanation facts.

```text
Matching result
→ xai_v1
→ deterministic pre-explanation
→ guarded narrative generation
→ explanation_v1
```

The explanation layer cannot change:

- MDMS score
- component scores
- weights
- evidence IDs
- missing-skill facts

Groq may be used for wording when configured; deterministic offline explanation remains available.

---

## 4. Tech Stack

### Core

- Python
- FastAPI
- Streamlit
- Pydantic / Pydantic Settings
- NumPy
- Pandas
- Scikit-learn

### NLP / ML

- PyTorch
- Transformers
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Cosine similarity
- Cross-Encoder experiments for Skill Matcher V2 research

### Document Processing

- Docling
- PyMuPDF / PyPDF
- Pillow
- OCR fallback adapters
- OpenCV / EasyOCR / RapidOCR / Tesseract where available

### LLM

- Groq API — optional
- Offline deterministic fallback

### Evaluation

- Recall@K
- nDCG@K
- Spearman correlation
- MAE
- Ablation studies
- Weight sensitivity analysis

---

## 5. Repository Structure

```text
.
├── app/
│   ├── api/                    # FastAPI routers, schemas and runtime services
│   └── frontend/               # Streamlit application
│
├── src/
│   ├── Data_loader/            # Document parsing, OCR and structured extraction
│   ├── Normalization/          # CV/JD normalization
│   ├── Representation/         # Features, chunks and embeddings
│   ├── Matching/               # Skill, Experience, Education, Semantic, MDMS
│   ├── Explainability/         # Evidence-grounded XAI
│   ├── Evaluation/             # Evaluation metrics / ground truth utilities
│   ├── Baselines/              # Baseline systems
│   ├── Optimization/           # Weight search / analysis
│   └── Experiments/            # Experimental research such as Skill V2
│
├── Data/
│   ├── Runtime/                # Runtime web artifacts
│   └── Results/                # Research / evaluation artifacts
│
├── configs/
│   └── mdms.yaml               # Matching/scoring configuration
│
├── scripts/                    # Offline experiment and utility scripts
├── tests/                      # Automated tests
├── docs/                       # Project documentation
├── requirements.txt
├── requirement_dev.txt
└── README.md
```

### Runtime vs Research Data

The project intentionally separates runtime state from research results:

```text
Data/Runtime/
    → active web/runtime artifacts

Data/Results/
    → experiments, metrics, benchmark outputs
```

Do not use `Data/Results` as runtime source-of-truth.

---

## 6. Installation

### 6.1 Clone the repository

```bash
git clone https://github.com/huycuong-2112/Intelligent-Recruitment-Assistant-Semantic-Resume-Screening-and-Job-Matching.git
cd Intelligent-Recruitment-Assistant-Semantic-Resume-Screening-and-Job-Matching
```

### 6.2 Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 6.3 Install project dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional development dependencies:

```bash
pip install -r requirement_dev.txt
```

### 6.4 Document-parser extras

The current parser code uses Docling directly and supports several optional OCR engines. If these packages are not yet included in your local `requirements.txt`, install the parser extras required by your environment:

```bash
pip install docling groq opencv-python easyocr pytesseract rapidocr-onnxruntime pdf2image
```

For PDF OCR through `pdf2image`, Windows users may also need **Poppler** installed and available on `PATH`.

For `pytesseract`, the Tesseract executable and relevant language packs must be installed separately.

> If your branch already pins these packages in `requirements.txt`, do not install them twice manually.

---

## 7. Environment Configuration

Create a local `.env` file in the repository root.

Example:

```env
GROQ_API_KEY=your_groq_api_key_here
```

`GROQ_API_KEY` is optional for the fully local fallback path.

Do **not** commit real API keys.

If no valid Groq key is available, the application should use the offline path where supported.

---

## 8. Run the Application

The local application uses **two processes**.

### Terminal 1 — FastAPI backend

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

### Terminal 2 — Streamlit frontend

```bash
python -m streamlit run app/frontend/Home.py
```

Open:

```text
http://localhost:8501
```

---

## 9. Typical UI Workflow

```text
1. Open Streamlit
2. Upload one or more CVs
3. Extract CV information
4. Review / Refine
5. Confirm selected CVs
6. Upload a Job Description
7. Extract JD information
8. Review / Refine
9. Confirm JD
10. Run Matching
11. Review candidate ranking
12. Generate / inspect candidate explanation
```

The system keeps CV and JD runtime identities separate and only matching-ready confirmed documents are consumed downstream.

---

## 10. API Overview

The canonical runtime API is versioned under:

```text
/api/v1
```

Key operations include:

```text
Resume parse / confirm
Job-description parse / confirm
Matching run
Explanation generation
```

When the backend is running, use FastAPI Swagger documentation for the exact current contract:

```text
http://127.0.0.1:8000/docs
```

---

## 11. Running Tests

Run the full regression suite:

```bash
python -m pytest -q
```

Run a specific test file:

```bash
python -m pytest -q tests/<test_file>.py
```

Development tooling is listed in:

```text
requirement_dev.txt
```

---

## 12. Evaluation

The project uses human-annotated ground truth for development evaluation.

Current evaluation metrics include:

```text
Recall@K
nDCG@K
Spearman correlation
MAE
```

Ground-truth relevance uses a 0–3 ordinal scale.

The current project research separates:

- development experiments
- blind evaluation

Blind data should remain untouched until model/scoring decisions are frozen.

---

## 13. Skill Matcher V2 Research

Production Skill Matcher V1 remains frozen while Skill Matcher V2 is evaluated experimentally.

Current research investigates:

- broader evidence pools
- requirement-level evidence retrieval
- composite requirement decomposition
- MiniLM retrieval
- Cross-Encoder evidence judging
- hard positive / hard negative benchmark construction

Experimental modules and artifacts must not silently modify production matching behavior.

---

## 14. Current Limitations

- Current MVP is local-only
- Docker/container packaging is not yet part of the locked runtime
- No cloud deployment is provided yet
- Current MDMS weights were selected on development data
- Blind evaluation is not yet claimed as final production evidence
- Skill Matcher V2 remains experimental
- Some OCR paths depend on external/system packages
- Semantic/evidence redundancy remains an active research question
- Education related-field taxonomy remains an active research question

---

## 15. Design Principles

This project follows several important constraints:

1. **Server-authoritative runtime state**  
   The frontend does not provide scores or evidence truth.

2. **No silent missing-value coercion**  
   `None`, `0.0`, and `NOT_APPLICABLE` have different meanings.

3. **Runtime / research separation**  
   Runtime web state is isolated from experiment artifacts.

4. **Evidence-grounded explainability**  
   Natural-language explanations cannot change deterministic scores or evidence identity.

5. **Reproducible experiments**  
   Production V1 is frozen while experimental versions are evaluated separately.

---

## 16. Security Notes

- Never commit `.env`
- Never commit `GROQ_API_KEY`
- API keys must remain backend-only
- Frontend requests must not contain secrets
- Runtime artifacts should not be treated as public benchmark data by default

---

## 17. Project Status

```text
Local MVP                 : Active
Streamlit UI              : Active
FastAPI backend           : Active
CV/JD extraction          : Active
Runtime confirmation      : Active
Normalization/embeddings  : Active
MDMS matching             : Active
Candidate ranking         : Active
XAI / candidate report    : Active
Skill Matcher V2          : Experimental
Docker / cloud deployment : Not implemented
```

---

## 18. Contributors

Add team members here:

```text
- <Name> — <Role / responsibility>
- <Name> — <Role / responsibility>
- <Name> — <Role / responsibility>
```

---

## 19. License

No license has been specified yet.

If the project will be published for reuse, add a `LICENSE` file and update this section.

---

## Citation / Academic Note

This repository is an academic/research prototype for intelligent recruitment assistance.  
The system is intended to support — not replace — human hiring decisions.
