"""
evaluate.py — Unified model evaluation: metrics, confusion matrix,
               ROC curve, PR curve, misclassified examples.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
from src.utils import PLOTS_DIR, REPORTS_DIR, log_info, log_success, log_warn


# ── Core Evaluation ────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    use_proba: bool = True,
    is_keras: bool = False,
) -> dict:
    """
    Compute and save all metrics, confusion matrix, ROC, PR curves.
    Returns dict of scalar metrics.
    """
    y_test = np.array(y_test)

    # ── Predictions ──────────────────────────────────────────────────────────
    if is_keras:
        y_prob = model.predict(X_test, batch_size=512, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        if use_proba and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif use_proba and hasattr(model, "decision_function"):
            df_scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-df_scores))   # sigmoid squash
        else:
            y_prob = None

    # ── Scalar Metrics ────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    log_info(
        f"{model_name:20s} | Acc={acc:.4f}  P={prec:.4f}  "
        f"R={rec:.4f}  F1={f1:.4f}"
        + (f"  AUC={roc:.4f}" if roc else "")
    )

    # ── Classification Report ─────────────────────────────────────────────────
    rpt_path = os.path.join(REPORTS_DIR, f"{model_name}_classification_report.txt")
    with open(rpt_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(classification_report(y_test, y_pred,
                                      target_names=["Negative", "Positive"]))
    log_success(f"Report → {rpt_path}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred, model_name)

    # ── ROC + PR Curves ───────────────────────────────────────────────────────
    if y_prob is not None:
        _plot_roc_curve(y_test, y_prob, model_name, roc)
        _plot_pr_curve(y_test, y_prob, model_name)

    return {
        "Model":     model_name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
        "ROC_AUC":   round(roc, 4) if roc else None,
    }


# ── Plot helpers ───────────────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Neg", "Pos"],
        yticklabels=["Neg", "Pos"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"Confusion matrix → {path}")


def _plot_roc_curve(y_true, y_prob, name: str, auc: float):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"ROC Curve — {name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"ROC curve → {path}")


def _plot_pr_curve(y_true, y_prob, name: str):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"AP = {ap:.4f}", color="darkorange")
    ax.set_title(f"Precision-Recall Curve — {name}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}_pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"PR curve → {path}")


# ── Comparison Visuals ─────────────────────────────────────────────────────────
def plot_metric_comparison(df: pd.DataFrame, metric: str):
    """Bar chart comparing models on a single metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    bars = ax.bar(df["Model"], df[metric], color=colors, edgecolor="black")
    for bar, val in zip(bars, df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_xlabel("Model"); ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"comparison_{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"{metric} comparison chart → {path}")


def save_comparison_table(df: pd.DataFrame):
    """Save results as CSV and as a matplotlib table image."""
    # CSV
    csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    log_success(f"Comparison CSV → {csv_path}")

    # Image table
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1",
            "ROC_AUC", "Training_Time_s"]
    sub = df[[c for c in cols if c in df.columns]].copy()

    fig, ax = plt.subplots(figsize=(14, max(3, len(sub) * 0.6 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=sub.values,
        colLabels=sub.columns,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    # Header styling
    for j in range(len(sub.columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Alternate row colors
    for i in range(1, len(sub) + 1):
        for j in range(len(sub.columns)):
            tbl[i, j].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    plt.title("Model Comparison Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    img_path = os.path.join(REPORTS_DIR, "model_comparison_table.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"Comparison table image → {img_path}")


def save_misclassified(
    texts_test, y_test, model, model_name: str,
    vectorizer=None, tokenizer=None,
    is_keras: bool = False,
    n: int = 20,
):
    """Save N misclassified examples to a CSV report."""
    y_test = np.array(y_test)

    if is_keras:
        from src.features import pad_seqs
        seqs   = tokenizer.texts_to_sequences(texts_test)
        X_test = pad_seqs(seqs)
        y_pred = (model.predict(X_test, batch_size=512, verbose=0).ravel() >= 0.5).astype(int)
    else:
        X_test = vectorizer.transform(texts_test)
        y_pred = model.predict(X_test)

    wrong_idx = np.where(y_pred != y_test)[0][:n]
    rows = []
    for i in wrong_idx:
        rows.append({
            "Review":    str(texts_test.iloc[i])[:300] if hasattr(texts_test, "iloc") else str(texts_test[i])[:300],
            "True":      "Positive" if y_test[i] == 1 else "Negative",
            "Predicted": "Positive" if y_pred[i] == 1 else "Negative",
        })
    df = pd.DataFrame(rows)
    path = os.path.join(REPORTS_DIR, f"{model_name}_misclassified.csv")
    df.to_csv(path, index=False)
    log_success(f"Misclassified examples ({len(rows)}) → {path}")
