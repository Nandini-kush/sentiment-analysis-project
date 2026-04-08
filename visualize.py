"""
visualize.py — Dataset-level visualizations:
  class distribution, review length, word clouds.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils import PLOTS_DIR, log_info, log_success, log_warn


def plot_class_distribution(df: pd.DataFrame):
    """Bar chart of positive vs negative counts."""
    counts = df["sentiment"].value_counts().sort_index()
    labels = {0: "Negative", 1: "Positive"}

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [labels[i] for i in counts.index],
        counts.values,
        color=["#e74c3c", "#2ecc71"],
        edgecolor="black"
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 100,
                f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14)
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"Class distribution → {path}")


def plot_review_length_distribution(df: pd.DataFrame):
    """Histogram + KDE of review word counts, split by class."""
    df = df.copy()
    df["length"] = df["review"].apply(lambda t: len(str(t).split()))

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "#e74c3c", "Negative"),
                                (1, "#2ecc71", "Positive")]:
        subset = df[df["sentiment"] == label]["length"]
        ax.hist(subset, bins=80, alpha=0.5, color=color,
                label=f"{name} (mean={subset.mean():.0f})")

    ax.set_xlim(0, 1000)
    ax.set_title("Review Length Distribution (word count)")
    ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "review_length_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log_success(f"Length distribution → {path}")


def plot_word_clouds(df: pd.DataFrame):
    """Word clouds for positive and negative reviews."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        log_warn("wordcloud not installed — skipping word cloud plots.")
        return

    for label, title, color in [
        (0, "Negative Reviews", "Reds"),
        (1, "Positive Reviews", "Greens"),
    ]:
        text = " ".join(
            df[df["sentiment"] == label]["review"].astype(str).tolist()
        )
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            colormap=color,
            max_words=200,
            collocations=False,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {title}", fontsize=16)
        plt.tight_layout()
        fname = title.lower().replace(" ", "_") + "_wordcloud.png"
        path  = os.path.join(PLOTS_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log_success(f"Word cloud → {path}")


def generate_all_eda(df: pd.DataFrame):
    """Run all dataset-level visualizations."""
    log_info("Generating EDA visualizations …")
    plot_class_distribution(df)
    plot_review_length_distribution(df)
    plot_word_clouds(df)
    log_success("EDA visualizations complete.")
