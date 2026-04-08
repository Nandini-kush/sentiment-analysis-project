"""
train_dl.py — Train deep learning models: ANN, LSTM, BiLSTM, GRU.
Uses TensorFlow/Keras with EarlyStopping and ModelCheckpoint.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd

from src.utils    import MODELS_DIR, PLOTS_DIR, log_info, log_success, log_warn, log_section, Timer, RANDOM_SEED
from src.features import (
    build_tokenizer, pad_seqs, save_tokenizer,
    transform_tfidf, build_tfidf, save_tfidf,
    DL_VOCAB_SIZE, DL_MAX_LEN
)
from src.evaluate import evaluate_model

try:
    import tensorflow as tf
    from tensorflow.keras.models   import Sequential, Model
    from tensorflow.keras.layers   import (
        Dense, Dropout, Embedding, LSTM, GRU,
        Bidirectional, GlobalAveragePooling1D,
        Conv1D, MaxPooling1D, BatchNormalization,
        Input, SpatialDropout1D,
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    log_warn("TensorFlow not installed — DL training skipped.")

EMBED_DIM  = 128
BATCH_SIZE = 256
MAX_EPOCHS = 20


def _model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, f"{name}.h5")


def _build_ann(vocab_size: int, max_len: int) -> "tf.keras.Model":
    """Shallow ANN on averaged embeddings."""
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, EMBED_DIM),
        GlobalAveragePooling1D(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ], name="ANN")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_lstm(vocab_size: int, max_len: int) -> "tf.keras.Model":
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, EMBED_DIM),
        SpatialDropout1D(0.2),
        LSTM(128, dropout=0.2, recurrent_dropout=0.1),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ], name="LSTM")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_bilstm(vocab_size: int, max_len: int) -> "tf.keras.Model":
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, EMBED_DIM),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.1)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ], name="BiLSTM")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_gru(vocab_size: int, max_len: int) -> "tf.keras.Model":
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, EMBED_DIM),
        SpatialDropout1D(0.2),
        Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.1)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ], name="GRU")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _get_callbacks(name: str):
    return [
        EarlyStopping(
            monitor="val_accuracy", patience=3,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            _model_path(name), monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-6, verbose=1
        ),
    ]


def _plot_history(history, name: str):
    """Save training curves to outputs/plots/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title(f"{name} — Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{name} — Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"Training curves → {path}")


def train_dl_models(
    texts_train, y_train,
    texts_val,   y_val,
    texts_test,  y_test,
) -> pd.DataFrame:
    """
    Full DL training pipeline.
    Returns results DataFrame.
    """
    if not TF_AVAILABLE:
        log_warn("TensorFlow unavailable — returning empty DL results.")
        return pd.DataFrame()

    log_section("Training Deep Learning Models")
    tf.random.set_seed(RANDOM_SEED)

    # Build tokenizer on train texts
    tokenizer, _ = build_tokenizer(texts_train, vocab_size=DL_VOCAB_SIZE)
    save_tokenizer(tokenizer)

    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_seqs(seqs, max_len=DL_MAX_LEN)

    X_train = encode(texts_train)
    X_val   = encode(texts_val)
    X_test  = encode(texts_test)

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    builders = {
        "ANN":    _build_ann,
        "LSTM":   _build_lstm,
        "BiLSTM": _build_bilstm,
        "GRU":    _build_gru,
    }

    results = []

    for name, builder in builders.items():
        log_info(f"\n── Training: {name} ──")
        model = builder(DL_VOCAB_SIZE, DL_MAX_LEN)
        model.summary(print_fn=lambda s: log_info(s))

        with Timer() as train_t:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=_get_callbacks(name),
                verbose=1,
            )
        log_success(f"Train time: {train_t}")
        _plot_history(history, name)

        # Evaluate on test set
        with Timer() as pred_t:
            metrics = evaluate_model(model, X_test, y_test, name,
                                     use_proba=True, is_keras=True)
        metrics["Training_Time_s"]   = round(train_t.elapsed, 3)
        metrics["Prediction_Time_s"] = round(pred_t.elapsed, 3)
        results.append(metrics)

    df = pd.DataFrame(results)
    log_section("DL Results")
    print(df[["Model", "Accuracy", "Precision", "Recall", "F1",
              "Training_Time_s"]].to_string(index=False))
    return df
