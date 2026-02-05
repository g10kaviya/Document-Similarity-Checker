# Document Similarity Checker (NLP Project)

This repository is a small Flask-based application for checking document similarity.
It extracts text from two uploaded PDFs, computes document-level similarity using
TF‑IDF and sentence-transformer embeddings (cosine), and highlights similar sentences.

Features
- Upload two PDF files via a simple web UI served by Flask (`backend/templates/upload.html`).
- Document-level TF‑IDF similarity and document embedding cosine similarity.
- Sentence-level detection using TF‑IDF and a sentence-transformer semantic similarity to highlight matching/"plagiarized" sentences.

Contents
- `backend/` – Flask app, templates, and upload folder.
- `backend/templates/upload.html` – simple single-page UI for uploading PDFs and showing results.
- `backend/utils/pdf_utils.py` – PDF text extraction helper (PyPDF2).
- `models/similarity.py` – similarity utilities (TF‑IDF, sentence splitting, sentence-transformer embedding similarity).

Requirements
- Python 3.9+ recommended.
- Key Python packages: Flask, PyPDF2, scikit-learn, spaCy, sentence-transformers, torch

Quick start (Windows PowerShell)
1. Create and activate a virtual environment in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install required packages:

```powershell
python -m pip install Flask PyPDF2 scikit-learn spacy sentence-transformers torch
python -m spacy download en_core_web_sm
```

3. Run the backend from the `backend` directory:

```powershell
cd backend
py app.py
# Open http://127.0.0.1:5000 in your browser
```

What the server returns
- `POST /upload` expects two files `file1` and `file2` (PDFs). It returns a JSON object with keys:
  - `similarity_score`: overall similarity (average of document embedding cosine and TF‑IDF similarity).
  - `cosine_similarity_score`: document-level embedding cosine (0..1).
  - `tfidf_similarity_score`: document-level TF‑IDF cosine (0..1).
  - `semantic_similarity_score`: document-level sentence-level semantic average (0..1).
  - `sentences`: array of extracted sentences from document 1.
  - `plagiarizedIndices`: indices of sentences flagged as similar by sentence-level checks.

Troubleshooting
- If you get import errors, make sure you installed the packages in the same Python environment used to run the app.
