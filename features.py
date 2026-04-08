"""
features.py — Feature engineering for ML and DL models.
  ML  : TF-IDF vectorization (with bigrams)
  DL  : Keras tokenization + sequence padding
"""

import os
import joblib
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.utils import (
    VEC_DIR, TOK_DIR, RANDOM_SEED, log_info, log_success
)

# ── Constants ──────────────────────────────────────────────────────────────────
TFIDF_PATH   = os.path.join(VEC_DIR, "tfidf_vectorizer.pkl")
TOKENIZER_PATH = os.path.join(TOK_DIR, "keras_tokenizer.pkl")

TFIDF_MAX_FEATURES = 50_000
DL_VOCAB_SIZE      = 20_000
DL_MAX_LEN         = 300


# ── TF-IDF ─────────────────────────────────────────────────────────────────────
def build_tfidf(texts_train, max_features: int = TFIDF_MAX_FEATURES):
    """
    Fit a TF-IDF vectorizer with unigrams and bigrams on training data.
    Returns (vectorizer, X_train_sparse).
    """
    log_info(f"Building TF-IDF (max_features={max_features:,}) …")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,          # log-scale TF
        strip_accents="unicode",
        analyzer="word",
        min_df=2,
    )
    X = vectorizer.fit_transform(texts_train)
    log_success(f"TF-IDF matrix shape: {X.shape}")
    return vectorizer, X


def transform_tfidf(vectorizer, texts):
    return vectorizer.transform(texts)


def save_tfidf(vectorizer):
    joblib.dump(vectorizer, TFIDF_PATH)
    log_success(f"TF-IDF vectorizer saved → {TFIDF_PATH}")


def load_tfidf():
    if not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError(f"TF-IDF not found at {TFIDF_PATH}. Train first.")
    v = joblib.load(TFIDF_PATH)
    log_success("TF-IDF vectorizer loaded.")
    return v


# ── Keras Tokenizer + Padding ──────────────────────────────────────────────────
def build_tokenizer(texts_train, vocab_size: int = DL_VOCAB_SIZE):
    """
    Fit a Keras Tokenizer and return (tokenizer, sequences_train).
    """
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except ImportError:
        raise ImportError("TensorFlow is required for DL feature extraction.")

    log_info(f"Building Keras Tokenizer (vocab={vocab_size:,}) …")
    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(texts_train)
    seqs = tok.texts_to_sequences(texts_train)
    log_success(f"Tokenizer fit. Vocab size: {len(tok.word_index):,}")
    return tok, seqs


def pad_seqs(sequences, max_len: int = DL_MAX_LEN):
    """Pad/truncate sequences to max_len."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def save_tokenizer(tok):
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tok, f)
    log_success(f"Keras tokenizer saved → {TOKENIZER_PATH}")


def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Train first.")
    with open(TOKENIZER_PATH, "rb") as f:
        tok = pickle.load(f)
    log_success("Keras tokenizer loaded.")
    return tok


# ── Train/Test Split ───────────────────────────────────────────────────────────
def split_data(texts, labels, test_size=0.2, val_size=0.1):
    """
    Split into train / val / test.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First split off test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
    )
    # Then split val from remaining
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_tmp
    )
    log_info(f"Split → Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test
