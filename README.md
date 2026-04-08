# 🎬 IMDB Sentiment Analysis — Final Year ML Project

> Multi-model sentiment classification on IMDB movie reviews with a desktop GUI.

---

## 🎯 Objective

Build, compare, and deploy multiple Machine Learning and Deep Learning models for binary sentiment classification (positive / negative) on the IMDB 50K review dataset.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Dataset** | IMDB 50K reviews (auto-extracted from zip) |
| **ML Models** | Logistic Regression, Naive Bayes, Linear SVM, Random Forest, XGBoost |
| **DL Models** | ANN, LSTM, BiLSTM, GRU |
| **Vectorisation** | TF-IDF (unigrams + bigrams, 50k features) |
| **DL Embedding** | Trainable embedding layer (128-dim) |
| **Visualisations** | 15+ auto-saved plots in `outputs/` |
| **Evaluation** | Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, ROC, PR curves |
| **Best Model** | Auto-selected by F1, saved to `models/` |
| **GUI** | Tkinter desktop app with probability bars |
| **CLI** | Full argument support via `main.py` |

---

## 📁 Project Structure

```
sentiment-analysis-project/
│
├── data/
│   └── IMDB Dataset.csv          ← put your dataset here
│
├── models/
│   ├── *.pkl                     ← trained ML models
│   ├── *.h5                      ← trained Keras models
│   ├── best_model_info.json      ← auto-generated
│   ├── vectorizers/              ← TF-IDF vectorizer
│   └── tokenizers/               ← Keras tokenizer
│
├── outputs/
│   ├── plots/                    ← all PNG charts
│   └── reports/                  ← CSVs + classification reports
│
├── src/
│   ├── data_loader.py            ← load & auto-detect dataset
│   ├── preprocess.py             ← text cleaning pipeline
│   ├── features.py               ← TF-IDF + Keras tokenization
│   ├── train_ml.py               ← ML model training
│   ├── train_dl.py               ← DL model training
│   ├── evaluate.py               ← metrics, plots, confusion matrices
│   ├── visualize.py              ← EDA visualizations
│   └── predict.py                ← inference module
│
├── gui.py                        ← Tkinter GUI
├── main.py                       ← CLI entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone / unzip the project

```bash
cd sentiment-analysis-project
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your dataset

Copy `IMDB Dataset.csv` **or** `TEXT.zip` into the `data/` folder.

```
data/
└── IMDB Dataset.csv   (or TEXT.zip — auto-extracted)
```

---

## 🚀 Running the Project

### Train everything (recommended first run)

```bash
python main.py --train-all
```

This will:
- Extract the dataset if needed
- Preprocess text
- Generate EDA visualizations
- Train all ML models (LR, NB, SVM, RF, XGBoost)
- Train all DL models (ANN, LSTM, BiLSTM, GRU)
- Save every model to `models/`
- Generate all comparison charts
- Save the best model to `best_model_info.json`

---

### Train only ML models (faster, no GPU needed)

```bash
python main.py --train-ml
```

### Train only DL models

```bash
python main.py --train-dl
```

### Generate visualizations only

```bash
python main.py --visualize
```

### Predict from terminal

```bash
python main.py --predict "The acting was brilliant and the plot kept me hooked!"
```

### Launch the GUI

```bash
python gui.py
```

---

## 📊 Expected Results

Typical accuracy on IMDB 50K:

| Model | Accuracy |
|---|---|
| Logistic Regression | ~90–92% |
| Linear SVM | ~90–92% |
| Naive Bayes | ~87–89% |
| Random Forest | ~84–87% |
| XGBoost | ~88–90% |
| BiLSTM | ~91–93% |
| GRU | ~91–93% |

---

## 🖼️ Screenshots

> *(Add your screenshots here after running)*

**GUI:**
```
[Screenshot of Tkinter prediction window]
```

**Comparison Chart:**
```
[Screenshot of outputs/plots/comparison_F1.png]
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: nltk` | `pip install nltk` |
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `No CSV found in data/` | Put `IMDB Dataset.csv` or `TEXT.zip` in `data/` |
| `best_model_info.json not found` | Run `python main.py --train-all` first |
| TensorFlow memory error | Reduce `BATCH_SIZE` in `src/train_dl.py` (line ~17) |
| Slow training on CPU | Use `--train-ml` only; DL training needs GPU for speed |
| `wordcloud` not found | `pip install wordcloud` (optional) |

---

## 🔮 Future Improvements

- Pre-trained embeddings (GloVe, FastText)
- BERT / DistilBERT fine-tuning
- Streamlit web app version
- Docker containerisation
- REST API (FastAPI)
- Attention visualisation
- Data augmentation

---

## 📋 Requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for DL)
- GPU optional but speeds up DL training 5–10×

---

## 👤 Author

*Final Year ML Project — IMDB Sentiment Analysis*
