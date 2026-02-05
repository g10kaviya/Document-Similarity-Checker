"""Flask application that exposes a simple UI and an `/upload` endpoint.

The `/upload` endpoint accepts two files (`file1` and `file2`) and returns a JSON
object containing similarity scores and sentence-level highlights. The overall
similarity returned by the current implementation is the average of the
document embedding cosine and the TF-IDF similarity.

See README.md for full usage instructions and API contract.
"""

from flask import Flask, render_template, request, jsonify

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pdf_utils import extract_text
from models.similarity import (
    similarity_score,
    sentence_level_plagiarism,
    sentence_level_semantic_plagiarism,
    document_embedding_cosine,
)
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Check both files are uploaded
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')
    if not file1 or not file2:
        return jsonify({"error": "Both files are required"}), 400
    
    # Save files locally
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
    file1.save(file1_path)
    file2.save(file2_path)

    # save files first
    text1 = extract_text(file1_path)
    text2 = extract_text(file2_path)

    # compute TF-IDF document similarity
    tfidf_score = similarity_score(text1, text2)

    # sentence-level TF-IDF and semantic similarity matrices + sentences
    tfidf_indices, sentences, tfidf_sim_matrix = sentence_level_plagiarism(text1, text2)
    semantic_indices, _, semantic_sim_matrix = sentence_level_semantic_plagiarism(text1, text2)

    # per-sentence maxima (for display)
    tfidf_sentence_scores = [float(max(row)) if len(row) else 0.0 for row in tfidf_sim_matrix]
    semantic_sentence_scores = [float(max(row)) if len(row) else 0.0 for row in semantic_sim_matrix]

    # document-level semantic score: average of per-sentence maxima
    semantic_score = sum(semantic_sentence_scores) / len(semantic_sentence_scores) if semantic_sentence_scores else 0.0

    # Document-level cosine similarity using a sentence-transformer (default model)
    cosine_score = document_embedding_cosine(text1, text2)

    # helper to clamp to [0,1]
    def _clamp01(v):
        try:
            v = float(v)
        except Exception:
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    cos_effective = _clamp01(cosine_score)
    tfidf_effective = _clamp01(tfidf_score)

    # overall is simple average of cosine and tfidf
    overall_score = (cos_effective + tfidf_effective) / 2.0

    combined_indices = list(set(tfidf_indices) | set(semantic_indices))

    return jsonify({
        "similarity_score": float(overall_score),
        "cosine_similarity_score": float(cos_effective),
        "tfidf_similarity_score": float(tfidf_effective),
        "semantic_similarity_score": float(semantic_score),
        "plagiarizedIndices": combined_indices,
        "sentences": sentences,
        "tfidf_sentence_scores": tfidf_sentence_scores,
        "semantic_sentence_scores": semantic_sentence_scores,
    })

if __name__ == "__main__":
    app.run(debug=True)
