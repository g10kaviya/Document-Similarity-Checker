"""Similarity utilities.

This module provides:
- document-level TF-IDF similarity (``similarity_score``)
- sentence-level TF-IDF and semantic (sentence-transformer) similarity
- utilities to load a custom trained SentenceTransformer from disk
"""

import os
import spacy
import logging
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


def similarity_score(text1, text2):
    """Document-level TF-IDF cosine similarity (0..1)."""
    vect = TfidfVectorizer().fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(vect)
    return float(similarity_matrix[0, 1])  # similarity score between documents

nlp = spacy.load("en_core_web_sm")

def split_to_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def sentence_level_plagiarism(text1, text2, threshold=0.7):
    sents1 = split_to_sentences(text1)
    sents2 = split_to_sentences(text2)
    vectorizer = TfidfVectorizer().fit(sents1 + sents2)
    vect1 = vectorizer.transform(sents1)
    vect2 = vectorizer.transform(sents2)
    sim_matrix = cosine_similarity(vect1, vect2)
    plagiarized_indices = []
    for i, row in enumerate(sim_matrix):
        if max(row) >= threshold:
            plagiarized_indices.append(i)
    return plagiarized_indices, sents1, sim_matrix


# default sentence-transformer used for sentence-level semantic checks
model = SentenceTransformer('all-mpnet-base-v2')


def sentence_level_semantic_plagiarism(text1, text2, threshold=0.7):
    sents1 = split_to_sentences(text1)  # Use spaCy or NLTK sentence splitter
    sents2 = split_to_sentences(text2)

    embeddings1 = model.encode(sents1, convert_to_tensor=True)
    embeddings2 = model.encode(sents2, convert_to_tensor=True)

    # Compute cosine similarity matrix between sentences
    sim_matrix = util.cos_sim(embeddings1, embeddings2).cpu().numpy()

    plagiarized_indices = []
    for i, row in enumerate(sim_matrix):
        if max(row) >= threshold:
            plagiarized_indices.append(i)

    return plagiarized_indices, sents1, sim_matrix


def document_embedding_cosine(text1, text2, model_name='all-mpnet-base-v2'):
    """Compute cosine similarity between two documents using a sentence-transformer model.

    Returns a float in [-1, 1] (usually near 0..1). Returns None on failure.
    """
    try:
        local_model = SentenceTransformer(model_name)
        emb1 = local_model.encode(text1, convert_to_tensor=True)
        emb2 = local_model.encode(text2, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        return float(score)
    except Exception:
        return None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _find_default_trained_model_dir():
    """Try common locations for a user-provided trained model directory.

    Returns the first existing directory or None. Also logs the candidates tried.
    """
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, 'models', 'trained_model'),
        os.path.join(cwd, 'models', 'custom_model'),
        os.path.join(cwd, 'trained_model'),
        os.path.join(cwd, 'model'),
        # common exported model from training runs (matches the path you provided)
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'model_epoch_3-20250923', 'model_epoch_3'),
    ]

    logger.info("Searching for trained model in candidate paths:")
    for p in candidates:
        logger.info("  trying: %s", p)
        if os.path.isdir(p):
            logger.info("Found trained model directory: %s", p)
            return p

    logger.info("No trained model directory found among candidates")
    return None


def load_custom_sentence_transformer(model_path=None):
    """Load a SentenceTransformer from a local directory if available.

    If model_path is None, try common locations. Returns the model or None.
    """
    try:
        # If explicit path provided and exists, try it first
        if model_path:
            logger.info("load_custom_sentence_transformer: explicit model_path provided: %s", model_path)
            if os.path.isdir(model_path):
                logger.info("Loading SentenceTransformer from explicit path: %s", model_path)
                return SentenceTransformer(model_path)
            else:
                logger.warning("Explicit model_path does not exist or is not a directory: %s", model_path)

        default = _find_default_trained_model_dir()
        if default:
            logger.info("Loading SentenceTransformer from default path: %s", default)
            return SentenceTransformer(default)
    except Exception as e:
        logger.error("Exception while loading custom SentenceTransformer model: %s", e)
        traceback.print_exc()
        return None

    return None


def trained_model_similarity(text1, text2, model_path=None):
    """Compute similarity with a user-supplied SentenceTransformer model directory.

    Returns a float in [-1,1] or None if no model found/loaded.
    """
    model = load_custom_sentence_transformer(model_path)
    if model is None:
        logger.info("trained_model_similarity: no model loaded (will return None)")
        return None
    try:
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2).item()
        return float(score)
    except Exception:
        return None