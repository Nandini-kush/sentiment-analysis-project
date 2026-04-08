"""
main.py — Entry point for the IMDB Sentiment Analysis project.

Usage:
  python main.py --train-all
  python main.py --train-ml
  python main.py --train-dl
  python main.py --evaluate
  python main.py --predict "This movie was absolutely fantastic!"
  python main.py --visualize
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils       import ensure_dirs, set_seeds, log_section, log_info, log_success, log_warn, log_error, save_best_model_info, MODELS_DIR
from src.data_loader import load_data, find_dataset
from src.preprocess  import preprocess_series
from src.features    import split_data
from src.visualize   import generate_all_eda
from src.evaluate    import plot_metric_comparison, save_comparison_table


def run_visualize(df: pd.DataFrame):
    log_section("EDA Visualizations")
    generate_all_eda(df)


def run_train_ml(df: pd.DataFrame) -> pd.DataFrame:
    from src.train_ml import train_ml_models

    log_section("ML Training Pipeline")
    texts = df["review"].values
    labels = df["sentiment"].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)
    results = train_ml_models(X_train, y_train, X_val, y_val, X_test, y_test)
    return results


def run_train_dl(df: pd.DataFrame) -> pd.DataFrame:
    from src.train_dl import train_dl_models

    log_section("DL Training Pipeline")
    texts  = df["review"].values
    labels = df["sentiment"].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)
    results = train_dl_models(X_train, y_train, X_val, y_val, X_test, y_test)
    return results


def determine_best_model(all_results: pd.DataFrame):
    """Pick model with highest F1 and save best_model_info.json."""
    if all_results.empty:
        log_warn("No results to determine best model.")
        return

    best_row = all_results.sort_values("F1", ascending=False).iloc[0]
    model_name = best_row["Model"]

    # Determine if it's DL or ML by checking file extensions
    ml_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    dl_path = os.path.join(MODELS_DIR, f"{model_name}.h5")

    if os.path.exists(dl_path):
        model_type = "dl"
    elif os.path.exists(ml_path):
        model_type = "ml"
    else:
        model_type = "ml"  # default fallback

    info = {
        "model_name":   model_name,
        "model_type":   model_type,
        "accuracy":     float(best_row.get("Accuracy", 0)),
        "precision":    float(best_row.get("Precision", 0)),
        "recall":       float(best_row.get("Recall", 0)),
        "f1":           float(best_row.get("F1", 0)),
        "roc_auc":      float(best_row.get("ROC_AUC", 0) or 0),
    }
    save_best_model_info(info)
    log_success(f"Best model: {model_name} (F1={info['f1']:.4f}, type={model_type})")


def main():
    parser = argparse.ArgumentParser(
        description="IMDB Sentiment Analysis — Training & Prediction"
    )
    parser.add_argument("--train-all",  action="store_true", help="Train ML + DL models")
    parser.add_argument("--train-ml",   action="store_true", help="Train ML models only")
    parser.add_argument("--train-dl",   action="store_true", help="Train DL models only")
    parser.add_argument("--visualize",  action="store_true", help="Generate EDA visualizations")
    parser.add_argument("--evaluate",   action="store_true", help="Show comparison charts (requires trained models)")
    parser.add_argument("--predict",    type=str,  default=None, help="Predict sentiment of a review string")
    parser.add_argument("--no-clean",   action="store_true", help="Skip text preprocessing (faster, lower accuracy)")

    args = parser.parse_args()

    # If no arguments given, default to --train-all
    if not any([args.train_all, args.train_ml, args.train_dl,
                args.visualize, args.evaluate, args.predict]):
        log_warn("No arguments provided. Defaulting to --train-all.")
        args.train_all = True

    # ── Setup ──────────────────────────────────────────────────────────────────
    ensure_dirs()
    set_seeds()

    # ── Predict only (no need to reload data) ──────────────────────────────────
    if args.predict:
        from src.predict import cli_predict
        cli_predict(args.predict)
        return

    # ── Load data ──────────────────────────────────────────────────────────────
    log_section("Loading Dataset")
    df = load_data()

    # ── Preprocess ─────────────────────────────────────────────────────────────
    if not args.no_clean:
        log_section("Text Preprocessing")
        df["review"] = preprocess_series(df["review"])

    # ── Visualize ──────────────────────────────────────────────────────────────
    if args.visualize or args.train_all:
        run_visualize(df)

    # ── Train ──────────────────────────────────────────────────────────────────
    all_results = pd.DataFrame()

    if args.train_ml or args.train_all:
        ml_results = run_train_ml(df)
        all_results = pd.concat([all_results, ml_results], ignore_index=True)

    if args.train_dl or args.train_all:
        dl_results = run_train_dl(df)
        all_results = pd.concat([all_results, dl_results], ignore_index=True)

    # ── Comparison charts ──────────────────────────────────────────────────────
    if not all_results.empty:
        log_section("Generating Comparison Charts")
        for metric in ["Accuracy", "Precision", "Recall", "F1"]:
            if metric in all_results.columns:
                plot_metric_comparison(all_results, metric)
        save_comparison_table(all_results)
        determine_best_model(all_results)

    elif args.evaluate:
        log_warn("No training results in memory. Load from CSV …")
        from src.utils import REPORTS_DIR
        csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
        if os.path.exists(csv_path):
            all_results = pd.read_csv(csv_path)
            for metric in ["Accuracy", "Precision", "Recall", "F1"]:
                if metric in all_results.columns:
                    plot_metric_comparison(all_results, metric)
            save_comparison_table(all_results)
        else:
            log_error("model_comparison.csv not found. Run training first.")

    log_section("Done")
    log_success("All tasks completed. Check outputs/ for results.")


if __name__ == "__main__":
    main()
