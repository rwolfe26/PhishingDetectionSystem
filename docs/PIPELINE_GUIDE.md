# Complete Email Phishing Detection Pipeline

This guide covers training, prediction, benchmarking, and tuning for the phishing detection pipeline.

## What It Does

The pipeline integrates three layers:

1. **Email Preprocessing** (42 numeric features)
   - URL analysis (IP addresses, HTTPS, suspicious ports, shorteners, homographs, brand Levenshtein)
   - Text analysis (word counts, character stats, urgency density)
   - Keyword analysis (urgency, credential, action, and auth keywords)
   - Header analysis (Reply-To / Return-Path / SPF / DKIM mismatches, received hops)
   - HTML analysis (forms, iframes, hidden text, external links)
   - Attachment analysis (executables, archives)
   - Phishing-specific signals (brand impersonation, sender display-name spoofing, generic greeting, all-caps subject)

2. **LSA Semantic Analysis** (configurable dimensions, default 128)
   - TF-IDF vectorisation with n-grams
   - Truncated SVD for dimensionality reduction
   - L2 normalisation for stability

3. **Random Forest Classifier**
   - Trained on combined feature vectors (42 numeric + N LSA dims)
   - Provides probability scores for risk assessment
   - `class_weight='balanced'` handles class imbalance automatically

## Results

Training on **17,204 emails** (11,306 ham + 5,898 phishing/spam) across 5 ham directories,
4 spam directories, `Phishing_Email.csv`, and a JSONL phishing corpus:

| Metric | Score |
|--------|-------|
| Accuracy | 98.55% |
| Precision | 98.04% |
| Recall | 97.71% |
| F1-Score | 97.88% |
| AUC-ROC | **99.91%** |
| False Positive Rate | 1.02% |
| False Negative Rate | 2.29% |

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Train the Pipeline

```bash
python run_pipeline.py --train
```

This will:
- Load all available ham/spam directories from `Datasets/`
- Load phishing emails from `Phishing_Email.csv` and the JSONL corpus
- Perform a single-pass preprocessing (parse → feature extraction → LSA fit → feature matrix)
- Train a Random Forest with `class_weight='balanced'`
- Evaluate on the held-out 20% test set
- Save models to `models/`

**Training options:**

```bash
python run_pipeline.py --train \
  --lsa-components 256 \      # LSA dimensions (default: 128)
  --lsa-min-df 2 \            # Minimum document frequency (default: 2)
  --lsa-max-df 0.85 \         # Maximum document frequency (default: 0.85)
  --test-size 0.2 \           # Test set proportion (default: 0.2)
  --csv-samples 3000 \        # Max phishing/safe samples from CSV (default: 1500)
  --cross-validate \          # Run stratified 5-fold CV before final training
  --cv-folds 5 \              # Number of CV folds (default: 5)
  --error-analysis            # Save misclassified emails to error_analysis.txt
```

### 2. Predict a Single Email

```bash
python run_pipeline.py --predict path/to/email.txt
```

Example output:
```
Prediction: SPAM
Confidence: 84.2%
Risk Level: MEDIUM (Possibly phishing)
```

Risk levels:
- **HIGH**: ≥ 85% phishing confidence
- **MEDIUM**: 60–85% phishing confidence
- **LOW**: 50–60% phishing confidence
- **SAFE**: < 50% phishing confidence

### 3. Benchmark Multiple Classifiers

Compares Random Forest, Logistic Regression, Linear SVM, Gradient Boosting, and XGBoost:

```bash
python run_pipeline.py --benchmark
```

### 4. Hyperparameter Tuning

Runs `RandomizedSearchCV` (30 iterations) over the Random Forest:

```bash
python run_pipeline.py --tune
```

### 5. LSA Dimensionality Search

Tests multiple LSA component counts and reports accuracy/AUC-ROC for each:

```bash
python experiments/lsa_dimension_search.py
python experiments/lsa_dimension_search.py --max-emails 3000  # faster run
```

Results are saved to `experiments/lsa_dimension_results.csv`.

### 6. Web Interface

Start the demo server and open http://localhost:8000:

```bash
uvicorn api.main:app --reload
```

API endpoints:
- `GET /health` — model status check
- `POST /classify` — classify raw email text (JSON)
- `POST /classify/file` — classify an uploaded `.eml` / `.txt` file

## Architecture

### File Structure

```
Detection_System/
├── run_pipeline.py               # Main CLI entry point
├── requirements.txt              # Python dependencies
│
├── pipeline/                     # Modular pipeline package
│   ├── __init__.py
│   ├── core.py                   # EmailPhishingPipeline (fit, extract, save, load)
│   ├── data_loader.py            # DataLoader (dirs, CSV, JSONL)
│   ├── trainer.py                # Trainer (train, CV, tune, benchmark, error analysis)
│   └── predictor.py              # Predictor (single email inference)
│
├── preprocessing/                # Email preprocessing
│   ├── __init__.py               # Public API + fit_lsa_encoder, fit_and_extract_features
│   ├── email_parser.py           # RFC 822 / MIME parsing
│   ├── url_extractor.py          # URL extraction and analysis
│   └── feature_extractor.py     # 42 numeric features
│
├── bert_base/                    # LSA semantic analysis
│   └── lsa_tool.py               # LSATextEncoder (TF-IDF + SVD + L2)
│
├── api/                          # FastAPI web application
│   ├── main.py                   # FastAPI app with /classify endpoints
│   ├── explainer.py              # Feature importance + SHAP + indicator highlighting
│   └── static/index.html         # Single-page web frontend
│
├── experiments/
│   └── lsa_dimension_search.py   # LSA component count experiment
│
├── models/                       # Saved trained models (after --train)
│   ├── lsa_encoder.pkl
│   ├── phishing_classifier.pkl
│   └── pipeline_metadata.pkl
│
├── Datasets/                     # Email datasets
│   ├── easy_ham/, easy_ham_2/, easy_ham_3/
│   ├── hard_ham/, hard_ham_2/
│   ├── spam/, spam_2/, spam_3/, spam_4/
│   ├── Phishing_Email.csv        # ~175k labelled phishing/safe emails
│   └── phishing and benign email dataset.jsonl
│
├── tests/
│   ├── test_preprocessing.py     # Preprocessing unit tests
│   └── test_pipeline.py          # Full pipeline integration tests (29 tests)
│
└── docs/                         # Documentation
```

### Feature List (42 numeric features)

**URL features (10):** `num_urls`, `num_unique_domains`, `has_ip_url`, `no_https_ratio`, `avg_url_length`, `max_url_length`, `avg_path_depth`, `total_dots_in_urls`, `has_at_symbol_url`, `has_suspicious_port`

**Text features (4):** `num_words`, `num_unique_words`, `num_chars`, `num_special_chars`

**Keyword features (3):** `num_urgent_keywords`, `num_credential_keywords`, `num_action_keywords`

**Header features (6):** `has_reply_to_mismatch`, `has_return_path_mismatch`, `num_received_hops`, `has_suspicious_mailer`, `subject_has_urgent`, `subject_has_re_fw`

**HTML features (6):** `has_html`, `has_form`, `has_iframe`, `has_hidden_text`, `num_external_links`, `link_text_url_mismatch`

**Attachment features (3):** `num_attachments`, `has_executable_attachment`, `has_archive_attachment`

**Phishing-specific signals (10):** `spf_dkim_fail`, `sender_domain_mismatch`, `num_homograph_chars`, `brand_impersonation_score`, `urgency_density`, `html_text_ratio`, `num_shortener_urls`, `greeting_generic`, `num_auth_keywords`, `subject_all_caps_ratio`

## Using the Pipeline Programmatically

```python
from pathlib import Path
from pipeline import EmailPhishingPipeline, DataLoader, Trainer, Predictor

# Load data
loader = DataLoader()
emails, labels = loader.load_dataset(ham_dirs, spam_dirs)
csv_emails, csv_labels = loader.load_phishing_csv(Path('Datasets/Phishing_Email.csv'), max_samples=2000)

# Train (single-pass: parse once, fit LSA, extract features)
pipeline = EmailPhishingPipeline(lsa_components=128)
X_train = pipeline.fit_lsa_and_extract(train_emails)
X_test = pipeline.extract_features(test_emails)

trainer = Trainer()
trainer.train_classifier(pipeline, X_train, y_train)
metrics = trainer.evaluate(pipeline, X_test, y_test)
pipeline.save_models(Path('models'))

# Predict
pipeline.load_models(Path('models'))
pred, prob = Predictor().predict_single(pipeline, raw_email_text)
```

## Explainability

Every `/classify` API response includes:
- **`top_features`** — the 8 numeric features that most influenced the prediction (RF importances × feature values)
- **`indicators`** — urgency phrases, credential keywords, action prompts, suspicious URLs found in the email text
- Optional **SHAP values** — pass `"use_shap": true` in the JSON request for SHAP TreeExplainer attributions

## Running Tests

```bash
python -m pytest tests/ -v
```

All 36 tests should pass.

## Troubleshooting

### "Model not loaded" on API startup
Run training first: `python run_pipeline.py --train`

### "Not enough LSA components" / ValueError after pruning
Reduce `--lsa-min-df` or increase `--csv-samples`.

### Slow training
- Reduce `--csv-samples` (default 1500 per class)
- Use `--lsa-components 64` for faster LSA
- All archives in `Datasets/` are already extracted — no further extraction needed

### Low accuracy
- Increase `--csv-samples` to include more true phishing examples
- Run `--tune` for hyperparameter search
- Run `--benchmark` to find the best classifier for your dataset
- Run `python experiments/lsa_dimension_search.py` to find the optimal LSA dimensions
