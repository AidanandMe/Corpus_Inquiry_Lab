# Corpus Inquiry Lab

A corpus-grounded research environment for **Digital Humanities and interpretive scholarship**, developed as an operational research instrument for collaborative inquiry (including **Linked Pasts 12 in Porto**).

The environment investigates how an AI-assisted research system represents the **boundaries of available evidence**, distinguishing between:
1. **Directly Supported Evidence** (literal text excerpts anchored to physical page coordinates);
2. **Derived Inference** (interpretive hypotheses and deductive bridges);
3. **Epistemic Uncertainty** (spatial proximity distribution and sparse retrieval signals);
4. **Conflicting Information / Competing Interpretations** (anti-harmonization and juxtaposition of tensions across sources);
5. **Information Absent from the Corpus** (transparent lexical findings distinguishing search limitations from conceptual absence).

---

## Architecture Overview

```
[ Active Corpus: Prepared or Uploaded ]
         │
         ├───▶ [ Deterministic Lexical Scanner ] ──▶ Exact Term Presence / Lexical Miss
         │                                              (Word-boundary matches, concordance context)
         │
         └───▶ [ Isolated Vector Store ] ────────▶ Dense Semantic Retrieval
                   (HNSW Cosine Space)                  (Proximity metrics, physical isolation)
                           │
                           ▼
               [ Epistemic Synthesis Engine ]
                   • Section 1: Directly Attributed Claims ([Card X] citations)
                   • Section 2: Inferred Connections (Hypotheses)
                   • Section 3: Divergent Perspectives (Competing accounts)
                   • Section 4: Epistemic Limitations (Search boundaries)
                   • Post-Generation Citation Audit
                           │
                           ▼
           [ Three-Panel Streamlit Workbench ]
```

---

## Directory Scaffolding

```
Corpus_Inquiry_Lab/
├── corpora/
│   ├── prepared/                    # Pre-packaged, generic research corpora
│   │   ├── sample_archaeological_survey/
│   │   └── sample_epigraphic_register/
│   └── session_uploads/             # Isolated temporary participant uploads
├── storage/                         # Physically isolated Chroma vector indices
│   ├── sample_archaeological_survey/
│   └── sample_epigraphic_register/
├── src/
│   ├── __init__.py
│   ├── config.py                    # Paths, chunk sizes, density thresholds
│   ├── ingest.py                    # PDF/text extraction, page provenance chunking
│   ├── lexical.py                   # Deterministic concordance scanner
│   ├── store.py                     # Per-corpus isolated vector storage & metrics
│   ├── retrieve.py                  # Coordinated unmerged dual retrieval
│   ├── synthesize.py                # Epistemic prompt engine & citation auditing
│   └── app.py                       # Three-Panel Streamlit web application
├── tests/
│   ├── test_ingest.py               # Extraction & page-boundary isolation tests
│   ├── test_retrieve.py             # Lexical miss, metrics, & isolation tests
│   ├── test_synthesize.py           # Prompt, section parsing, & anti-coherence tests
│   └── test_app.py                  # Application discovery & import smoke tests
├── requirements.txt
└── README.md
```

---

## Quickstart Guide

### 1. Prerequisites & Virtual Environment
The project runs on standard Python 3.11+. The local virtual environment is installed in `.venv`.

To activate and verify:
```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/ -v
```

### 2. Running the Application
Launch the turnkey Streamlit workbench:
```powershell
.\.venv\Scripts\streamlit run src/app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## Methodological Safeguards Implemented

1. **Similarity is Not Confidence**:
   All vector proximity scores are strictly labeled as mathematical metrics (`Cosine Proximity: 0.XXXX`). The system never labels them as confidence, certainty, probability, or evidential strength.
2. **Transparent, Deterministic Lexical Searching**:
   Zero LLMs are involved in determining search terms. The scanner operates on explicit terms supplied by the researcher or extracted via a documented rule (preserving quoted phrases and filtering high-frequency stopwords).
3. **Lexical Miss vs. Conceptual Absence**:
   A zero lexical match establishes strictly that *"The searched term or phrase was not found in the corpus using this lexical check."* It does not assert that the concept is absent, acknowledging that historical variants, translations, or synonyms may exist.
4. **Anti-Harmonization (Prevention of Artificial Coherence)**:
   The synthesis engine is forbidden from smoothing over conflicting accounts. Disagreements in dates, locations, or interpretations must be juxtaposed in Section 3 (*Divergent or Contrasting Perspectives*).
5. **Physical and Logical Corpus Isolation**:
   Each corpus has its own dedicated directory under `storage/<corpus_id>/` and separate collection. Retrieval against Corpus A is physically incapable of returning chunks belonging to Corpus B.
6. **Reproducible Embeddings**:
   Embeddings use the local `all-MiniLM-L6-v2 (ONNX)` model (384-dimensional). Model identity, metric, chunk count, and timestamp are preserved in each corpus's `index_metadata.json`.
