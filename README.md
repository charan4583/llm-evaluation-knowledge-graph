# LLM Evaluation & Knowledge Graph Explorer

An end-to-end Python AI engineering application for evaluating Large Language Model (LLM) responses against reference answers, analyzing sentence-level claims, extracting entities and relationships, and exploring interactive knowledge graphs.

---

## Problem Statement

Evaluating LLM-generated responses is challenging because two valid answers can convey the same meaning using different phrasing, while fluent responses can omit key facts or introduce unsupported statements. Relying solely on exact string matching misses semantic equivalence, whereas relying solely on an LLM judge lacks deterministic mathematical transparency.

This project combines:
- **Deterministic NLP & Information Retrieval Metrics**: Precision, Recall, F1-score, Exact Match, and Keyword Coverage.
- **Dense Semantic Embeddings**: Cosine similarity using `sentence-transformers/all-MiniLM-L6-v2`.
- **Claim-Level Verification**: Sentence-by-sentence comparison against reference evidence (`Supported`, `Partially supported`, `Requires verification`, `Not found in reference`).
- **Optional Gemini AI Judge**: Structured JSON evaluation validated with Pydantic (core features work completely offline without an API key).
- **Knowledge Graph Construction & Analytics**: Entity extraction (spaCy NER + domain rules), directed relationship extraction, NetworkX graph metrics, and interactive PyVis visualization.
- **Document Processing, Persistence & Reporting**: PDF (`PyMuPDF`), DOCX (`python-docx`), and TXT ingestion; SQLite + SQLAlchemy persistence; CSV, JSON, and formatted PDF (`ReportLab`) exports.

---

## Architecture

```
llm-evaluation-knowledge-graph/
├── app.py                          # Main Streamlit entrypoint
├── config/settings.py              # Environment-based configuration
├── pages/                          # 8-page modular Streamlit UI
│   ├── 1_Dashboard.py              # Live analytics from SQLite
│   ├── 2_Evaluate_Response.py      # Single-response evaluation & claim analysis
│   ├── 3_Compare_Responses.py      # Multi-response benchmarking (2-5 models)
│   ├── 4_Knowledge_Graph.py        # Interactive PyVis + NetworkX explorer
│   ├── 5_Text_Analysis.py          # Single-text stats & two-text comparison
│   ├── 6_History.py                # Searchable SQLite evaluation history
│   ├── 7_Reports.py                # CSV, JSON, and PDF report downloads
│   └── 8_Settings.py               # API status, thresholds, data management
├── src/
│   ├── evaluation/                 # Precision, Recall, F1, Exact Match, Semantic Similarity
│   ├── nlp/                        # Normalization, tokenization, NER, relationship extraction
│   ├── graph/                      # NetworkX builder, centrality analytics, PyVis HTML
│   ├── ai/                         # Gemini client, Pydantic AI Judge, claim analyzer
│   ├── database/                   # SQLAlchemy ORM models, connection, repository
│   ├── document_processing/        # PDF, DOCX, TXT validation and extraction
│   ├── export/                     # CSV, JSON, and ReportLab PDF exporters
│   └── ui/                         # Dark/light theme CSS and shared UI helpers
├── tests/                          # Automated Pytest suite
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.12+ (compatible with 3.11+) |
| **Web UI** | Streamlit, Plotly |
| **NLP & Embeddings** | spaCy (`en_core_web_sm`), Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Knowledge Graph** | NetworkX, PyVis |
| **Optional AI Judge** | Google Gemini API (`google-genai`), Pydantic |
| **Database** | SQLite, SQLAlchemy 2.x |
| **Document Ingestion & Export** | PyMuPDF (`fitz`), `python-docx`, ReportLab, Pandas |
| **Testing** | Pytest |

---

## Evaluation Methodology

- **Precision, Recall & F1-Score**: Reference and generated texts are normalized (lowercased, punctuation stripped, whitespace collapsed), tokenized, filtered for stopwords, and lightly lemmatized. Using multiset token counts:
  - `Precision = TP / (TP + FP)` — fraction of generated content tokens present in the reference.
  - `Recall = TP / (TP + FN)` — fraction of reference content tokens captured.
  - `F1 = 2 * Precision * Recall / (Precision + Recall)` — harmonic mean of Precision and Recall.
  - Division by zero is handled safely (returns `0.0`).
- **Exact Match**: Returns `1.0` (100%) when normalized reference and generated strings are identical, `0.0` otherwise.
- **Keyword Coverage**: Extracts top frequency-ranked content keywords from the reference answer and computes the fraction matched in the generated answer.
- **Semantic Similarity**: Encodes both answers with `sentence-transformers/all-MiniLM-L6-v2` and computes cosine similarity. *Note: Semantic similarity measures closeness of meaning in embedding space, not factual truth.*
- **Knowledge Graph**: Extracts entities via spaCy NER + domain rules, extracts directed relationships grounded in source sentences, builds a `networkx.DiGraph`, and computes degree centrality, connected components, and isolated nodes.

---

## Installation & Setup

### 1. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure environment variables (optional)
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY if you want to use the optional AI Judge
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Running Automated Tests

```bash
python -m pytest -v
```

---

## Limitations & Future Improvements

- **Lexical Baseline**: Token-overlap Precision/Recall/F1 penalizes valid synonyms; semantic similarity helps bridge this gap, though neither proves universal factual truth.
- **Scanned PDFs**: PyMuPDF extracts embedded text only; scanned image-based PDFs without OCR text layers are flagged clearly rather than silently failing.
- **Future Improvements**: Add OCR support (`pytesseract`), sentence-level NLI cross-encoder models for deeper claim verification, and batch dataset CSV evaluation.
