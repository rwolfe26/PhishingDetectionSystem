---
title: Phishing Detector
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Email Phishing Detection System

A production-ready machine learning system that classifies emails as phishing or legitimate in real time. Built end-to-end — from raw feature engineering and model training to a deployed web application with a live demo.

**[Try the live demo →](https://rwolfe26-phishing-detector.hf.space)**

---

## What it does

Paste any email (or upload a `.txt` / `.eml` file) and the system:

- Classifies it as **phishing** or **safe** with a confidence score
- Gives a **plain-English summary** of why — written for non-technical users, not data scientists
- Explains *why* using the top contributing features and SHAP values
- Highlights the specific words and phrases that triggered the detection
- Assigns a risk level: `HIGH / MEDIUM / LOW / SAFE`

---

## Model performance

Trained on **17,204 emails** (11,306 legitimate + 5,898 phishing/spam) across 5 public datasets:

| Metric | Score |
|--------|-------|
| Accuracy | **98.55%** |
| Precision | 98.04% |
| Recall | 97.71% |
| F1-Score | 97.88% |
| AUC-ROC | **99.91%** |
| False Positive Rate | 1.02% |
| False Negative Rate | 2.29% |

---

## Technical highlights

### Feature engineering — 175 features total

Rather than relying solely on a pre-trained language model, I built a 47-feature handcrafted extractor that captures email-specific signals a generic model would miss:

| Category | Features | What it catches |
|----------|----------|-----------------|
| URL analysis | 10 | IP-based URLs, HTTPS ratio, shorteners (bit.ly etc.), brand typosquatting via Levenshtein distance |
| Text analysis | 7 | Word count, special-char ratio, unique-word ratio, caps ratio |
| Keyword analysis | 3 | Urgency phrases, credential requests, action words |
| Header analysis | 6 | Reply-To mismatch, SPF/DKIM failure, display-name spoofing |
| HTML analysis | 6 | Hidden forms, iframes, invisible text |
| Attachments | 3 | Executable extensions, double-extension tricks (`.pdf.exe`) |
| Phishing signals | 10 | Brand impersonation score, homograph characters, urgency density, generic greetings |
| URL redirect resolution | 2 | Follows shortened URLs to their final destination domain |

These 47 numeric features are combined with **128-dimensional LSA semantic embeddings** (TF-IDF → Truncated SVD → L2 normalisation) to produce a **175-feature vector** per email.

### Machine learning

- **Random Forest** (200 trees, balanced class weights) — chosen for its robustness to outliers and interpretability
- Benchmarked against XGBoost, Logistic Regression, Linear SVM, and Gradient Boosting
- Hyperparameter tuning via `RandomizedSearchCV` (30 parameter combinations)
- 5-fold cross-validation with per-fold error analysis

### Explainability

- **Plain-English summary** — rule-based narrative explanation generated from all 44 numeric signals; written for non-technical users (e.g. "The sender's display name doesn't match their actual email address, a classic impersonation tactic")
- **Feature importance** — ranked contribution of each of the 175 features
- **SHAP values** — per-prediction Shapley explanations
- **Indicator highlighting** — the exact words/phrases flagged in the email text

### Production system

| Component | Technology |
|-----------|-----------|
| REST API | FastAPI + Uvicorn, structured JSON logging, request-ID middleware |
| Frontend | Vanilla JS SPA — split-panel layout, Barlow Condensed + JetBrains Mono typography, plain-English result summaries |
| Browser extension | Chrome extension for Gmail and Outlook Web — injects a real-time risk badge into email threads as you read them (see below) |
| IMAP Monitor | Polls a live mailbox, classifies incoming emails, stores results in SQLite |
| Dashboard | Live monitoring view — classification history, stats, detection rate |
| Feedback loop | Users correct wrong predictions; corrections feed back into retraining |
| Testing | 60+ pytest tests across pipeline, preprocessing, API, and monitor |
| CI/CD | GitHub Actions — lint (ruff) + full test suite on every push |
| Deployment | Docker → Hugging Face Spaces (16 GB RAM), models stored on HF Hub |

---

## Browser Extension

The repository includes a Chrome extension (`browser-extension/`) that brings phishing detection directly into Gmail and Outlook Web. When you open an email, the extension automatically classifies it and injects a colour-coded risk badge at the top of the thread — no copy-pasting required.

**Features:**
- Works on Gmail (`mail.google.com`) and Outlook Web (`outlook.live.com`, `outlook.office.com`)
- Colour-coded `HIGH / MEDIUM / LOW / SAFE` badge with confidence score
- Expandable plain-English explanation powered by the same API
- Configurable API endpoint via the extension popup (defaults to the hosted HF Space)

**Status:** The extension is fully functional and available for local use. It is not currently listed on the Chrome Web Store as this project is in active development. To try it yourself:

```bash
# 1. Clone the repo
git clone https://github.com/rwolfe26/PhishingDetectionSystem

# 2. Open Chrome and go to chrome://extensions
# 3. Enable Developer mode (top-right toggle)
# 4. Click "Load unpacked" and select the browser-extension/ folder
```

The extension will immediately start scanning emails in Gmail and Outlook Web, hitting the live API at `https://rwolfe26-phishing-detector.hf.space`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Web Frontend (SPA)                  │
│          Paste email / Upload file / View result        │
└────────────────────────┬────────────────────────────────┘
                         │ POST /classify
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                      │
│  • Feature extraction (47 numeric)                      │
│  • LSA encoding (128 dims)                              │
│  • Random Forest inference                              │
│  • SHAP + feature importance explanation                │
└──────┬──────────────────────────────────────┬───────────┘
       │                                      │
┌──────▼───────┐                   ┌──────────▼──────────┐
│  ML Pipeline │                   │   IMAP Monitor      │
│  core.py     │                   │   monitor.py        │
│  trainer.py  │                   │   SQLite storage    │
│  predictor.py│                   │   Dashboard API     │
└──────────────┘                   └─────────────────────┘
```

---

## Project structure

```
├── pipeline/               # ML pipeline (train, predict, benchmark, tune)
│   ├── core.py             # EmailPhishingPipeline — orchestrates everything
│   ├── data_loader.py      # Loads from dirs, CSV, JSONL
│   ├── trainer.py          # Training, cross-validation, hyperparameter tuning
│   └── predictor.py        # Single-email inference
│
├── preprocessing/          # Email feature extraction
│   ├── email_parser.py     # RFC 822 / MIME parsing
│   ├── url_extractor.py    # URL analysis + redirect resolution
│   └── feature_extractor.py  # 47 handcrafted numeric features
│
├── bert_base/lsa_tool.py   # TF-IDF + TruncatedSVD semantic encoder
│
├── api/                    # FastAPI web application
│   ├── main.py             # REST endpoints + middleware
│   ├── explainer.py        # SHAP + feature importance + indicator highlighting
│   └── static/             # Dark-mode SPA + monitoring dashboard
│
├── email_monitor/          # IMAP monitoring system
│   ├── imap_client.py      # IMAP connection + unseen email fetcher
│   ├── monitor.py          # Classification loop + alerting
│   ├── storage.py          # SQLite classification history
│   └── feedback.py         # User correction store (active learning)
│
├── tests/                  # 60+ pytest tests
├── run_pipeline.py         # CLI entry point
├── monitor.py              # IMAP monitor CLI entry point
├── download_models.py      # Downloads models from HF Hub at startup
└── Dockerfile              # Production container
```

---

## Running locally

```bash
git clone https://github.com/rwolfe26/PhishingDetectionSystem
cd PhishingDetectionSystem
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download pre-trained models
python download_models.py  # set HF_REPO_ID=rwolfe26/phishing-detector first
# or train from scratch (requires datasets in Datasets/)
python run_pipeline.py --train

# Start the API
uvicorn api.main:app --reload
# Open http://localhost:8000
```

---

## Training pipeline CLI

```bash
# Train with cross-validation and error analysis
python run_pipeline.py --train --cross-validate --error-analysis

# Benchmark all classifiers
python run_pipeline.py --benchmark

# Hyperparameter tuning
python run_pipeline.py --tune

# Retrain including user feedback corrections
python run_pipeline.py --train --include-feedback

# Classify a single email file
python run_pipeline.py --predict path/to/email.txt
```

---

## Tech stack

**Python** · **scikit-learn** · **FastAPI** · **SHAP** · **SQLite** · **Docker** · **GitHub Actions** · **Hugging Face Hub/Spaces**

---

## License

MIT
