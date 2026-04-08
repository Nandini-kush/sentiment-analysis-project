"""
data_loader.py — Load, inspect, and auto-detect IMDB CSV columns.
Also handles automatic ZIP extraction.
"""

import os
import zipfile
import pandas as pd
from src.utils import (
    DATA_DIR, log_info, log_success, log_warn
)

# Preferred column names (case-insensitive search)
PREFERRED_TEXT_COLS = ["review", "text", "comment", "sentence", "content"]
PREFERRED_LABEL_COLS = ["sentiment", "label", "target", "class", "category"]


def extract_zip_if_needed(zip_path: str, dest_dir: str = DATA_DIR) -> str | None:
    """
    If a zip exists at zip_path, extract it into dest_dir.
    Returns the path to the extracted CSV file, or None if not found.
    """
    if not os.path.exists(zip_path):
        return None

    log_info(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    log_success(f"Extracted to {dest_dir}")

    # Find the CSV inside
    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(root, f)
                log_success(f"Found CSV: {csv_path}")
                return csv_path

    return None


def find_dataset() -> str:
    """
    Search for a CSV dataset in the data/ directory.
    Handles nested zip extraction automatically.
    Returns the CSV file path.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # 1. Look for any CSV directly
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".csv"):
            return os.path.join(DATA_DIR, f)

    # 2. Try extracting a ZIP
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(".zip"):
            csv_path = extract_zip_if_needed(os.path.join(DATA_DIR, f))
            if csv_path:
                return csv_path

    raise FileNotFoundError(
        f"No CSV (or extractable ZIP) found in '{DATA_DIR}'. "
        "Place 'IMDB Dataset.csv' or 'TEXT.zip' there and retry."
    )


def _detect_column(df: pd.DataFrame, preferred: list[str], kind: str) -> str:
    """Return the best matching column name (case-insensitive)."""
    cols_lower = {str(c).lower(): c for c in df.columns}

    for pref in preferred:
        if pref.lower() in cols_lower:
            return cols_lower[pref.lower()]

    # Fallback: pick first string-like column for text, last for label
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not str_cols:
        raise ValueError(f"Cannot detect {kind} column. Columns: {list(df.columns)}")

    chosen = str_cols[0] if kind == "text" else str_cols[-1]
    log_warn(f"Auto-detected {kind} column: '{chosen}'")
    return chosen


def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load the IMDB dataset CSV and normalize it to two columns:
      'review'    — raw review text
      'sentiment' — 0 (negative) or 1 (positive)
    """
    if csv_path is None:
        csv_path = find_dataset()

    log_info(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    log_success(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Detect columns
    text_col = _detect_column(df, PREFERRED_TEXT_COLS, "text")
    label_col = _detect_column(df, PREFERRED_LABEL_COLS, "label")

    df = df[[text_col, label_col]].copy()
    df.columns = ["review", "sentiment"]

    # Drop nulls / empty reviews
    before = len(df)
    df.dropna(subset=["review", "sentiment"], inplace=True)
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"] != ""].copy()
    log_info(f"Dropped {before - len(df)} null/empty rows -> {len(df):,} remain")

    # Encode labels robustly
    raw_labels = df["sentiment"].astype(str).str.lower().str.strip()

    mapping = {
        "positive": 1,
        "pos": 1,
        "1": 1,
        "true": 1,
        "good": 1,
        "negative": 0,
        "neg": 0,
        "0": 0,
        "false": 0,
        "bad": 0,
    }

    unmapped = sorted(set(raw_labels.unique()) - set(mapping.keys()))
    if unmapped:
        raise ValueError(f"Unknown label values found: {unmapped}")

    df["sentiment"] = raw_labels.map(mapping).astype(int)
    log_info("Label mapping applied successfully: positive/negative -> 1/0")

    # Class balance
    counts = df["sentiment"].value_counts().sort_index()
    log_info(f"Class distribution:\n{counts.to_string()}")

    return df