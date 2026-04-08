"""
gui.py — Tkinter GUI for IMDB Sentiment Prediction.

Launch: python gui.py
Requirements: Train models first with `python main.py --train-all`
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import ensure_dirs

# ── Colours & Fonts ────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
SURFACE   = "#16213e"
CARD      = "#0f3460"
ACCENT    = "#e94560"
POSITIVE  = "#00b894"
NEGATIVE  = "#d63031"
TEXT      = "#edf2f4"
SUBTEXT   = "#b2bec3"
FONT_HEAD = ("Segoe UI", 22, "bold")
FONT_BODY = ("Segoe UI", 11)
FONT_MONO = ("Consolas", 11)
FONT_BTN  = ("Segoe UI", 12, "bold")
FONT_SMLL = ("Segoe UI", 9)


class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_dirs()

        self.title("IMDB Sentiment Analyser")
        self.geometry("820x680")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(680, 560)

        self._predict_fn = None   # lazy-loaded predictor
        self._build_ui()
        self._load_model_async()

    # ── UI Construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=CARD, padx=20, pady=12)
        header.pack(fill="x")

        tk.Label(header, text="🎬  IMDB Sentiment Analyser",
                 font=FONT_HEAD, bg=CARD, fg=TEXT).pack(side="left")

        self.status_lbl = tk.Label(
            header, text="⏳ Loading model…",
            font=FONT_SMLL, bg=CARD, fg=SUBTEXT
        )
        self.status_lbl.pack(side="right")

        # ── Main content ─────────────────────────────────────────────────────
        content = tk.Frame(self, bg=BG, padx=24, pady=18)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        # Input label
        tk.Label(content, text="Paste or type your movie review:",
                 font=FONT_BODY, bg=BG, fg=SUBTEXT, anchor="w"
                 ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        # Text area
        self.text_area = scrolledtext.ScrolledText(
            content,
            font=FONT_MONO,
            bg=SURFACE, fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            padx=12, pady=10,
            wrap="word",
            height=10,
        )
        self.text_area.grid(row=1, column=0, sticky="nsew")
        self.text_area.insert("1.0", "Enter your review here…")
        self.text_area.bind("<FocusIn>",  self._on_focus_in)
        self.text_area.bind("<FocusOut>", self._on_focus_out)
        self._placeholder_active = True

        # Button row
        btn_row = tk.Frame(content, bg=BG)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        self.predict_btn = tk.Button(
            btn_row, text="  Analyse Sentiment  ",
            font=FONT_BTN,
            bg=ACCENT, fg="white",
            activebackground="#c0392b",
            relief="flat", padx=20, pady=8,
            cursor="hand2",
            command=self._on_predict,
        )
        self.predict_btn.pack(side="left")

        self.clear_btn = tk.Button(
            btn_row, text="  Clear  ",
            font=FONT_BTN,
            bg=SURFACE, fg=SUBTEXT,
            activebackground=CARD,
            relief="flat", padx=16, pady=8,
            cursor="hand2",
            command=self._on_clear,
        )
        self.clear_btn.pack(side="left", padx=(10, 0))

        # ── Result card ────────────────────────────────────────────────────
        self.result_frame = tk.Frame(content, bg=SURFACE, padx=20, pady=18)
        self.result_frame.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        self.result_frame.columnconfigure(0, weight=1)

        self.result_lbl = tk.Label(
            self.result_frame, text="—",
            font=("Segoe UI", 28, "bold"),
            bg=SURFACE, fg=SUBTEXT
        )
        self.result_lbl.grid(row=0, column=0, sticky="w")

        self.confidence_lbl = tk.Label(
            self.result_frame, text="",
            font=("Segoe UI", 12),
            bg=SURFACE, fg=SUBTEXT
        )
        self.confidence_lbl.grid(row=1, column=0, sticky="w", pady=(4, 12))

        # Probability bars
        bar_frame = tk.Frame(self.result_frame, bg=SURFACE)
        bar_frame.grid(row=2, column=0, sticky="ew")
        bar_frame.columnconfigure(1, weight=1)

        for i, (label, colour, attr) in enumerate([
            ("Positive", POSITIVE, "pos_bar"),
            ("Negative", NEGATIVE, "neg_bar"),
        ]):
            tk.Label(bar_frame, text=label, font=FONT_SMLL,
                     bg=SURFACE, fg=SUBTEXT, width=9, anchor="w"
                     ).grid(row=i, column=0, sticky="w")

            canvas = tk.Canvas(bar_frame, height=18, bg=CARD,
                               highlightthickness=0)
            canvas.grid(row=i, column=1, sticky="ew", padx=(6, 6))

            pct_lbl = tk.Label(bar_frame, text="", font=FONT_SMLL,
                               bg=SURFACE, fg=SUBTEXT, width=8, anchor="e")
            pct_lbl.grid(row=i, column=2, sticky="e")

            setattr(self, attr, (canvas, colour, pct_lbl))

        bar_frame.bind("<Configure>", lambda e: self._redraw_bars())

        # Footer
        tk.Label(
            self, text="Train models first: python main.py --train-all",
            font=FONT_SMLL, bg=BG, fg=SUBTEXT
        ).pack(side="bottom", pady=6)

        self._pos_pct = 0.0
        self._neg_pct = 0.0

    # ── Placeholder handling ───────────────────────────────────────────────────
    def _on_focus_in(self, _event):
        if self._placeholder_active:
            self.text_area.delete("1.0", "end")
            self.text_area.configure(fg=TEXT)
            self._placeholder_active = False

    def _on_focus_out(self, _event):
        if not self.text_area.get("1.0", "end").strip():
            self.text_area.insert("1.0", "Enter your review here…")
            self.text_area.configure(fg=SUBTEXT)
            self._placeholder_active = True

    # ── Async model loading ────────────────────────────────────────────────────
    def _load_model_async(self):
        def _load():
            try:
                from src.predict import predict_review
                # Warm-up
                predict_review("test")
                self._predict_fn = predict_review
                self.after(0, lambda: self.status_lbl.configure(
                    text="✅ Model ready", fg=POSITIVE
                ))
            except FileNotFoundError:
                self.after(0, lambda: self.status_lbl.configure(
                    text="⚠ No model found — train first", fg=NEGATIVE
                ))
            except Exception as e:
                self.after(0, lambda: self.status_lbl.configure(
                    text=f"⚠ {str(e)[:60]}", fg=NEGATIVE
                ))

        threading.Thread(target=_load, daemon=True).start()

    # ── Predict ────────────────────────────────────────────────────────────────
    def _on_predict(self):
        review = self.text_area.get("1.0", "end").strip()
        if not review or self._placeholder_active or review == "Enter your review here…":
            messagebox.showwarning("Empty Input", "Please enter a review before predicting.")
            return

        if self._predict_fn is None:
            messagebox.showerror(
                "Model Not Ready",
                "Model is still loading or not trained.\n"
                "Run: python main.py --train-all"
            )
            return

        self.predict_btn.configure(state="disabled", text="Analysing…")
        self.result_lbl.configure(text="…", fg=SUBTEXT)
        self.confidence_lbl.configure(text="")
        self.update()

        def _run():
            try:
                result = self._predict_fn(review)
                self.after(0, lambda: self._show_result(result))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.predict_btn.configure(
                    state="normal", text="  Analyse Sentiment  "
                ))

        threading.Thread(target=_run, daemon=True).start()

    def _show_result(self, result: dict):
        label    = result["label"]
        conf     = result["confidence"]
        pos_pct  = result["positive_prob"]
        neg_pct  = result["negative_prob"]

        colour = POSITIVE if label == "Positive" else NEGATIVE
        emoji  = "😊" if label == "Positive" else "😞"

        self.result_lbl.configure(
            text=f"{emoji}  {label}",
            fg=colour
        )
        self.confidence_lbl.configure(
            text=f"Confidence: {conf:.1f}%",
            fg=colour
        )

        self._pos_pct = pos_pct / 100.0
        self._neg_pct = neg_pct / 100.0
        self._update_bar_labels(pos_pct, neg_pct)
        self._redraw_bars()

    def _update_bar_labels(self, pos_pct, neg_pct):
        self.pos_bar[2].configure(text=f"{pos_pct:.1f}%")
        self.neg_bar[2].configure(text=f"{neg_pct:.1f}%")

    def _redraw_bars(self):
        for (canvas, colour, _), pct in [
            (self.pos_bar, self._pos_pct),
            (self.neg_bar, self._neg_pct),
        ]:
            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            filled = int(w * pct)
            canvas.create_rectangle(0, 0, filled, h, fill=colour, outline="")

    # ── Clear ──────────────────────────────────────────────────────────────────
    def _on_clear(self):
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", "Enter your review here…")
        self.text_area.configure(fg=SUBTEXT)
        self._placeholder_active = True

        self.result_lbl.configure(text="—", fg=SUBTEXT)
        self.confidence_lbl.configure(text="")
        self._pos_pct = self._neg_pct = 0.0
        self._update_bar_labels(0, 0)
        self._redraw_bars()


def main():
    app = SentimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
