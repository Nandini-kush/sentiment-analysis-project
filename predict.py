"""
predict.py — Load the best saved model and predict a single review.
Works from CLI: python main.py --predict "Your review here"
"""

import os
import numpy as np
import joblib

from src.utils import log_info, log_success, log_error, load_best_model_info
from src.preprocess import clean_text


def _load_ml_model(model_name: str):
    from src.utils import MODELS_DIR
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _load_keras_model(model_name: str):
    from src.utils import MODELS_DIR
    import tensorflow as tf
    path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path)


def predict_review(review_text: str) -> dict:
    """
    Load the best model and predict the sentiment of a single review.
    Returns dict with 'label', 'confidence', 'positive_prob', 'negative_prob'.
    """
    info = load_best_model_info()
    model_name = info["model_name"]
    model_type = info.get("model_type", "ml")   # "ml" or "dl"

    log_info(f"Using best model: {model_name} (type={model_type})")

    cleaned = clean_text(review_text, remove_stopwords=True, stem=False)

    if model_type == "ml":
        from src.features import load_tfidf
        vectorizer = load_tfidf()
        model      = _load_ml_model(model_name)
        X = vectorizer.transform([cleaned])

        pred   = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            pos_p = float(probs[1])
        else:
            pos_p = float(pred)

    elif model_type == "dl":
        from src.features import load_tokenizer, pad_seqs, DL_MAX_LEN
        tokenizer = load_tokenizer()
        model     = _load_keras_model(model_name)
        seqs  = tokenizer.texts_to_sequences([cleaned])
        X     = pad_seqs(seqs, max_len=DL_MAX_LEN)
        pos_p = float(model.predict(X, verbose=0)[0][0])
        pred  = int(pos_p >= 0.5)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    label      = "Positive" if pred == 1 else "Negative"
    confidence = pos_p if pred == 1 else (1 - pos_p)

    return {
        "label":         label,
        "confidence":    round(confidence * 100, 2),
        "positive_prob": round(pos_p * 100, 2),
        "negative_prob": round((1 - pos_p) * 100, 2),
    }


def cli_predict(review_text: str):
    """Print a formatted prediction result to terminal."""
    try:
        result = predict_review(review_text)
        print("\n" + "═" * 50)
        print(f"  Review   : {review_text[:100]}{'…' if len(review_text) > 100 else ''}")
        print(f"  Sentiment: {result['label']}")
        print(f"  Confidence  : {result['confidence']:.1f}%")
        print(f"  Positive ▶  {result['positive_prob']:.1f}%")
        print(f"  Negative ▶  {result['negative_prob']:.1f}%")
        print("═" * 50 + "\n")
    except Exception as e:
        log_error(f"Prediction failed: {e}")
