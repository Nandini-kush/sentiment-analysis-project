"""
utils.py — Shared utilities: logging, paths, seeds, JSON helpers.
"""

import os
import json
import random
import time
import numpy as np
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

VEC_DIR     = os.path.join(MODELS_DIR, "vectorizers")
TOK_DIR     = os.path.join(MODELS_DIR, "tokenizers")
ENC_DIR     = os.path.join(MODELS_DIR, "encoders")

BEST_MODEL_JSON = os.path.join(MODELS_DIR, "best_model_info.json")

RANDOM_SEED = 42


def ensure_dirs():
    """Create all required project directories."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR,
              VEC_DIR, TOK_DIR, ENC_DIR]:
        os.makedirs(d, exist_ok=True)


def set_seeds(seed: int = RANDOM_SEED):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ── Logging helpers ────────────────────────────────────────────────────────────
def log_info(msg: str):
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL}  {msg}")

def log_success(msg: str):
    print(f"{Fore.GREEN}[OK]{Style.RESET_ALL}    {msg}")

def log_warn(msg: str):
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL}  {msg}")

def log_error(msg: str):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

def log_section(title: str):
    bar = "─" * 60
    print(f"\n{Fore.MAGENTA}{bar}")
    print(f"  {title}")
    print(f"{bar}{Style.RESET_ALL}\n")


# ── Timing ─────────────────────────────────────────────────────────────────────
class Timer:
    """Simple context-manager timer."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


# ── JSON helpers ───────────────────────────────────────────────────────────────
def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log_success(f"Saved JSON → {path}")


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ── Model registry ─────────────────────────────────────────────────────────────
def save_best_model_info(info: dict):
    save_json(info, BEST_MODEL_JSON)

def load_best_model_info() -> dict:
    if not os.path.exists(BEST_MODEL_JSON):
        raise FileNotFoundError(
            "best_model_info.json not found. Run training first."
        )
    return load_json(BEST_MODEL_JSON)
