"""
preprocess.py — Text cleaning and preprocessing pipeline.
Handles HTML stripping, lowercasing, punctuation, stopwords.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.utils import log_info, log_success, log_warn

# Download NLTK data silently
def _download_nltk():
    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            log_info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)

_download_nltk()

# Compile regex patterns once for speed
_HTML_RE  = re.compile(r"<[^>]+>")
_URL_RE   = re.compile(r"https?://\S+|www\.\S+")
_BR_RE    = re.compile(r"<br\s*/?>", re.IGNORECASE)
_NONALPHA = re.compile(r"[^a-zA-Z\s]")
_SPACES   = re.compile(r"\s+")

STOP_WORDS = set(stopwords.words("english"))
# Keep a few sentiment-bearing negations
KEEP_WORDS = {"not", "no", "nor", "never", "neither", "nothing", "nobody",
              "nowhere", "none", "cannot", "couldn't", "wouldn't", "shouldn't",
              "isn't", "wasn't", "aren't", "weren't", "hasn't", "haven't",
              "hadn't", "doesn't", "don't", "didn't"}
STOP_WORDS -= KEEP_WORDS

_stemmer = PorterStemmer()


def clean_text(
    text: str,
    remove_stopwords: bool = True,
    stem: bool = False,
) -> str:
    """
    Full text cleaning pipeline:
      1. Remove HTML tags and <br> tags
      2. Remove URLs
      3. Lowercase
      4. Remove non-alphabetic characters
      5. Collapse whitespace
      6. Optionally remove stopwords (keeping negations)
      7. Optionally stem
    """
    # 1. HTML
    text = _BR_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    # 2. URLs
    text = _URL_RE.sub(" ", text)
    # 3. Lowercase
    text = text.lower()
    # 4. Non-alpha
    text = _NONALPHA.sub(" ", text)
    # 5. Whitespace
    text = _SPACES.sub(" ", text).strip()

    if remove_stopwords or stem:
        tokens = text.split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]
        if stem:
            tokens = [_stemmer.stem(t) for t in tokens]
        text = " ".join(tokens)

    return text


def preprocess_series(
    series: pd.Series,
    remove_stopwords: bool = True,
    stem: bool = False,
    show_progress: bool = True,
) -> pd.Series:
    """
    Apply clean_text to an entire pandas Series with optional progress display.
    """
    log_info(f"Preprocessing {len(series):,} texts …")
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Cleaning")
        cleaned = series.progress_apply(
            lambda t: clean_text(t, remove_stopwords, stem)
        )
    except ImportError:
        cleaned = series.apply(
            lambda t: clean_text(t, remove_stopwords, stem)
        )
    log_success("Preprocessing complete.")
    return cleaned


def get_review_lengths(series: pd.Series) -> pd.Series:
    """Return word-count per review (raw, before cleaning)."""
    return series.apply(lambda t: len(str(t).split()))
