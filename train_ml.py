"""
train_ml.py — Train and evaluate traditional ML models.
Models: Logistic Regression, Naive Bayes, Linear SVM,
        Random Forest, XGBoost (optional).
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.naive_bayes     import MultinomialNB
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.calibration     import CalibratedClassifierCV

from src.utils    import MODELS_DIR, log_info, log_success, log_warn, log_section, Timer
from src.features import (
    build_tfidf, transform_tfidf, save_tfidf, split_data
)
from src.evaluate import evaluate_model

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    log_warn("XGBoost not installed — skipping XGBoost model.")


def _save_ml_model(model, name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    log_success(f"Model saved → {path}")
    return path


def get_ml_models():
    """Return dict of {name: estimator}."""
    models = {
        "Logistic_Regression": LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="saga",
            random_state=42,
            n_jobs=-1,
        ),
        "Naive_Bayes": MultinomialNB(alpha=0.1),
        "Linear_SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42),
            cv=3,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    return models


def train_ml_models(
    texts_train, y_train,
    texts_val,   y_val,
    texts_test,  y_test,
    run_cv: bool = True,
) -> pd.DataFrame:
    """
    Full ML training pipeline.
    Fits TF-IDF, trains every model, evaluates, returns results DataFrame.
    """
    log_section("Training ML Models")

    # Build TF-IDF
    vectorizer, X_train = build_tfidf(texts_train)
    X_val   = transform_tfidf(vectorizer, texts_val)
    X_test  = transform_tfidf(vectorizer, texts_test)
    save_tfidf(vectorizer)

    models  = get_ml_models()
    results = []

    for name, model in models.items():
        log_info(f"\n── Training: {name} ──")

        # Train
        with Timer() as train_t:
            model.fit(X_train, y_train)
        log_success(f"Train time: {train_t}")

        # Optional CV on combined train+val
        if run_cv:
            import scipy.sparse as sp
            X_combined = sp.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            cv_scores  = cross_val_score(
                model, X_combined, y_combined,
                cv=3, scoring="f1", n_jobs=-1
            )
            log_info(f"CV F1 (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Evaluate on test set
        with Timer() as pred_t:
            metrics = evaluate_model(model, X_test, y_test, name, use_proba=True)
        log_success(f"Predict time: {pred_t}")

        metrics["Training_Time_s"] = round(train_t.elapsed, 3)
        metrics["Prediction_Time_s"] = round(pred_t.elapsed, 3)
        results.append(metrics)

        # Save model
        _save_ml_model(model, name)

    df = pd.DataFrame(results)
    log_section("ML Results")
    print(df[["Model", "Accuracy", "Precision", "Recall", "F1",
              "Training_Time_s"]].to_string(index=False))
    return df
