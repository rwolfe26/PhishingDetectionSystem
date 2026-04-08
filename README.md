# Email Phishing Detection System

A machine learning pipeline for detecting phishing emails using handcrafted feature engineering and LSA semantic analysis.

## Quick Start

```bash
# Install dependencies
source .venv/bin/activate
pip install -r requirements.txt

# Train the model (uses all available datasets automatically)
python run_pipeline.py --train

# Classify an email
python run_pipeline.py --predict path/to/email.txt

# Start the web demo
uvicorn api.main:app --reload
# Open http://localhost:8000
```

## Current Performance

Training on **17,204 emails** (11,306 ham + 5,898 phishing/spam):

| Metric | Score |
|--------|-------|
| Accuracy | 98.55% |
| Precision | 98.04% |
| Recall | 97.71% |
| F1-Score | 97.88% |
| AUC-ROC | **99.91%** |
| False Positive Rate | 1.02% |
| False Negative Rate | 2.29% |

## Project Structure

```
Detection_System/
├── run_pipeline.py               # CLI: --train, --predict, --benchmark, --tune
├── requirements.txt
├── Dockerfile                    # Container image for deployment
├── .github/workflows/ci.yml      # GitHub Actions: pytest + ruff on every push
│
├── pipeline/                     # Modular ML pipeline
│   ├── core.py                   # EmailPhishingPipeline (default 128 LSA dims)
│   ├── data_loader.py            # Load emails from dirs, CSV, JSONL
│   ├── trainer.py                # Train, CV, hyperparameter tune, benchmark
│   └── predictor.py              # Single-email inference
│
├── preprocessing/                # Email feature extraction
│   ├── email_parser.py           # RFC 822 / MIME parsing
│   ├── url_extractor.py          # URL analysis + URLRedirectResolver
│   └── feature_extractor.py     # 44 numeric features
│
├── bert_base/
│   └── lsa_tool.py               # TF-IDF + TruncatedSVD semantic encoder
│
├── api/                          # FastAPI web application
│   ├── main.py                   # REST API with JSON logging + request-ID middleware
│   ├── explainer.py              # Feature importance + SHAP + indicators
│   └── static/index.html         # Dark-mode SPA frontend
│
├── experiments/
│   └── lsa_dimension_search.py   # Find optimal LSA component count
│
├── models/                       # Saved trained models (run --train to generate)
├── Datasets/                     # Email datasets (extracted archives)
├── tests/                        # 54 pytest tests
│   ├── test_pipeline.py          # Pipeline, preprocessing, explainer (29 tests)
│   ├── test_preprocessing.py     # Email parser, URL extractor (7 tests)
│   └── test_api.py               # FastAPI endpoints + single-pass training (18 tests)
└── docs/                         # Documentation
```

## Features

### Email Preprocessing (44 Features)

| Category | Count | Key signals |
|----------|-------|-------------|
| URL analysis | 10 | IP URLs, HTTPS ratio, shorteners (bit.ly etc.), brand Levenshtein distance |
| Text analysis | 4 | Word count, character stats |
| Keyword analysis | 3 | Urgency, credential, action keywords |
| Header analysis | 6 | Reply-To mismatch, SPF/DKIM fail, sender display-name spoofing |
| HTML analysis | 6 | Forms, iframes, hidden text |
| Attachments | 3 | Executables, archives |
| Phishing signals | 10 | Brand impersonation score, homograph chars, urgency density, generic greeting |
| URL redirect resolution | 2 | Final domain after redirect, total redirect hops |

### LSA Semantic Analysis

- **TF-IDF Vectorisation** with bigrams
- **Truncated SVD** — 128 dimensions by default (tunable via `--lsa-components`)
- **L2 Normalisation** for length-independent embeddings
- Combined vector: 44 numeric + 128 LSA = **172 total features**

### Classification

- **Random Forest** — 200 trees, `class_weight='balanced'`
- **Classifier benchmarking** — compare RF, XGBoost, Logistic Regression, Linear SVM, Gradient Boosting
- **Hyperparameter tuning** — `RandomizedSearchCV` over 30 param combinations
- **Explainability** — top contributing features + SHAP values + phishing indicator highlighting

## CLI Reference

```bash
# Full training with cross-validation and error analysis
python run_pipeline.py --train --cross-validate --error-analysis

# Training with more phishing data from CSV
python run_pipeline.py --train --csv-samples 5000

# Benchmark classifiers (RF vs XGBoost vs SVM vs LR)
python run_pipeline.py --benchmark

# Hyperparameter tuning
python run_pipeline.py --tune

# LSA dimension search
python experiments/lsa_dimension_search.py --max-emails 3000

# Predict with explain (via API)
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"email_text": "...", "use_shap": false}'
```

## Testing

```bash
python -m pytest tests/ -v    # 54 tests
```

## Documentation

| Document | Contents |
|----------|---------|
| [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) | Full CLI reference, architecture, feature list |
| [docs/LSA_PREPROCESSING_INTEGRATION.md](docs/LSA_PREPROCESSING_INTEGRATION.md) | LSA integration, single-pass training, API |
| [docs/MODULAR_STRUCTURE.md](docs/MODULAR_STRUCTURE.md) | Module design and class overview |
| [docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md) | Project evolution from monolithic to modular |
| [bert_base/lsa_research_report.md](bert_base/lsa_research_report.md) | Why LSA over BERT |

## Dataset Information

All archives in `Datasets/` are pre-extracted and ready to use:

| Source | Emails | Label |
|--------|--------|-------|
| SpamAssassin easy_ham + easy_ham_2 + easy_ham_3 | 8,954 | Ham |
| SpamAssassin hard_ham + hard_ham_2 | 752 | Ham |
| SpamAssassin spam + spam_2 + spam_3 + spam_4 | 4,298 | Spam |
| Phishing_Email.csv (1,500 per class by default) | 3,000 | Mixed |
| JSONL phishing corpus | 200 | Mixed |

Additional CSV/ZIP datasets in `Datasets/` are available but not currently used by the default pipeline.

## API Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins (e.g. `https://myapp.com`) |
| `MODEL_DIR` | `./models` | Path to directory containing trained `.pkl` model files |

## Deployment

**Local (uvicorn):**
```bash
uvicorn api.main:app --reload
```

**Docker:**
```bash
docker build -t phishing-detector .
# Mount your trained models and run
docker run -p 8000:8000 -v $(pwd)/models:/app/models phishing-detector
```

## Roadmap

### Phase 1 (Complete)
- [x] 44-feature engineering pipeline (42 → 44 with redirect resolution features)
- [x] Multi-dataset training (17k+ emails)
- [x] Classifier benchmarking + hyperparameter tuning
- [x] Cross-validation + error analysis
- [x] FastAPI backend + web frontend
- [x] Feature importance + SHAP explainability
- [x] URL redirect resolver (`URLRedirectResolver` — follows shorteners to final domain)
- [x] Enhanced attachment analysis (double-extension detection)
- [x] Structured JSON logging + request-ID middleware
- [x] CI pipeline (GitHub Actions) + Dockerfile

### Phase 2 (Planned)
- [ ] Gmail / IMAP integration (read-only OAuth)
- [ ] Real-time classification API with streaming
- [ ] User feedback loop for active learning
- [ ] Monitoring dashboard

### Phase 3 (Future)
- [ ] Multi-language support
- [ ] Adversarial robustness testing
- [ ] Federated learning

## License

[Add your license here]
