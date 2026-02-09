# Email Phishing Detection System

A production-ready machine learning pipeline for detecting phishing emails using preprocessing and semantic analysis.

## Quick Start

### Installation
```bash
# Clone the repository
cd Detection_System

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (already installed)
pip install scikit-learn joblib numpy
```

### Train the Model
```bash
# Basic training
python run_pipeline.py --train

# Custom parameters
python run_pipeline.py --train --lsa-components 256 --test-size 0.2
```

### Classify Emails
```bash
# Predict a single email
python run_pipeline.py --predict Datasets/spam/0000.7b1b73cf36cf9dbc3d64e3f2ee2b91f1

# Example output:
# Prediction: SPAM
# Confidence: 62.85%
# Risk Level: LOW (Possibly spam)
```

## 📊 Current Performance

Training on **3,052 emails** (2,551 ham + 501 spam):
- **Accuracy: 99.35%**
- **Precision: 98.98%**
- **Recall: 97.00%**
- **F1-Score: 97.98%**
- **False Positive Rate: 0.20%** (1 in 511 ham emails)
- **False Negative Rate: 3.00%** (3 in 100 spam emails)

## Project Structure

```
Detection_System/
├── run_pipeline.py              # Main CLI entry point
│
├── pipeline/                     # Modular pipeline package
│   ├── __init__.py              # Package exports
│   ├── core.py                  # EmailPhishingPipeline class
│   ├── data_loader.py           # Data loading utilities
│   ├── trainer.py               # Training and evaluation
│   └── predictor.py             # Prediction logic
│
├── preprocessing/                # Email preprocessing module
│   ├── __init__.py              # LSA integration
│   ├── email_parser.py          # RFC 822 email parsing
│   ├── url_extractor.py         # URL analysis and extraction
│   └── feature_extractor.py    # 34 numeric features
│
├── bert_base/                    # Semantic analysis
│   ├── lsa_tool.py              # LSA encoder (768 dimensions)
│   └── lsa_research_report.md   # Methodology documentation
│
├── models/                       # Saved trained models
│   ├── lsa_encoder.pkl          # Trained LSA encoder
│   ├── phishing_classifier.pkl  # Random Forest classifier
│   └── pipeline_metadata.pkl    # Pipeline configuration
│
├── Datasets/                     # Email datasets
│   ├── easy_ham/                # Legitimate emails (2,551)
│   ├── spam/                    # Spam emails (501)
│   └── *.tar.bz2                # Additional datasets (compressed)
│
├── tests/                        # Unit tests
│   └── test_preprocessing.py    # Preprocessing tests
│
├── docs/                         # Documentation
│   ├── MODULAR_STRUCTURE.md     # Detailed module docs
│   ├── PIPELINE_GUIDE.md        # Pipeline usage guide
│   ├── LSA_PREPROCESSING_INTEGRATION.md  # Integration details
│   └── REFACTORING_SUMMARY.md   # Refactoring notes
│
├── examples/                     # Example scripts
│   └── example_lsa_preprocessing_integration.py
│
├── old_models/                   # Archived models
│   └── pipeline_monolithic_backup.py  # Original pipeline
│
└── README.md                     # This file
```

## Features

### Email Preprocessing (34 Features)
- **URL Analysis**: IP addresses, HTTPS ratio, suspicious ports, path depth
- **Text Analysis**: Word counts, special characters, unique words
- **Header Analysis**: Reply-To mismatches, suspicious mailers, received hops
- **HTML Analysis**: Forms, iframes, hidden text, external links
- **Attachment Analysis**: Executables, archives, attachment count

### LSA Semantic Analysis (256-768 Dimensions)
- **TF-IDF Vectorization**: Sublinear scaling with n-grams
- **Truncated SVD**: Latent semantic structure extraction
- **L2 Normalization**: Stable embeddings independent of email length

### Classification
- **Random Forest**: 100 decision trees, optimized hyperparameters
- **Feature Importance**: Identifies key phishing indicators
- **Risk Levels**: HIGH/MEDIUM/LOW/Safe based on confidence scores

## Documentation

Detailed documentation is available in the `docs/` directory:

- **[MODULAR_STRUCTURE.md](docs/MODULAR_STRUCTURE.md)** - Module architecture and design
- **[PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)** - Complete usage guide
- **[LSA_PREPROCESSING_INTEGRATION.md](docs/LSA_PREPROCESSING_INTEGRATION.md)** - Integration details
- **[REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md)** - Project evolution

## Usage Examples

### Programmatic Usage

```python
from pathlib import Path
from pipeline import EmailPhishingPipeline, DataLoader, Trainer, Predictor

# Initialize and train
pipeline = EmailPhishingPipeline(lsa_components=256)
data_loader = DataLoader()
emails, labels = data_loader.load_dataset(ham_dirs, spam_dirs)

pipeline.fit_lsa(emails)
X = pipeline.extract_features(emails)

trainer = Trainer()
trainer.train_classifier(pipeline, X_train, y_train)
pipeline.save_models(Path('models'))

# Load and predict
pipeline.load_models(Path('models'))
predictor = Predictor()
prediction, probability = predictor.predict_single(pipeline, email_text)
```

### Integration

```python
from preprocessing import preprocess_email_with_lsa
import joblib

# Load saved models
lsa_encoder = joblib.load('models/lsa_encoder.pkl')
classifier = joblib.load('models/phishing_classifier.pkl')

# Process email
result = preprocess_email_with_lsa(raw_email, lsa_encoder)
X = result['combined_vector'].reshape(1, -1)

# Predict
prediction = classifier.predict(X)[0]  # 0=ham, 1=spam
probability = classifier.predict_proba(X)[0][1]  # spam confidence
```

## 🎯 Roadmap & Requirements

### MVP (Current Status) ✅
- ✅ Email parsing and feature extraction
- ✅ LSA semantic analysis
- ✅ Random Forest classifier (99.35% accuracy)
- ✅ CLI interface for training and prediction
- ✅ Model persistence and loading
- ✅ Risk level classification

### Phase 1 (Planned)
- [ ] Web interface (FastAPI + React)
- [ ] Explainability (SHAP/LIME feature attribution)
- [ ] Highlighted spans showing phishing indicators
- [ ] URL expansion and redirect resolution
- [ ] Enhanced attachment analysis

### Phase 2 (Future)
- [ ] Gmail/IMAP integration (read-only OAuth)
- [ ] Real-time classification API
- [ ] User feedback loop for active learning
- [ ] Dashboard for monitoring and analytics
- [ ] A/B testing for model improvements

### Phase 3 (Extended)
- [ ] Multi-language support
- [ ] Brand impersonation detection
- [ ] Adversarial robustness testing
- [ ] Federated learning for privacy
- [ ] Mobile app integration

## Technical Details

### Feature Engineering
- **Header Analysis**: From/Reply-To domain matching, X-Mailer patterns
- **URL Forensics**: Entropy, TLD rarity, homograph detection, redirect chains
- **Body Analysis**:
  - Lexical: Urgency keywords, financial terms, credential requests
  - Structural: HTML forms, hidden text, base64 blobs, CSS tricks
- **Semantic**: Transformer-inspired embeddings via LSA

### Model Architecture
```
Raw Email
    ↓
[Email Parser] → Subject, Body, Headers, Attachments
    ↓
[URL Extractor] → URL patterns and analysis
    ↓
[Feature Extractor] → 34 numeric features
    ↓
[LSA Encoder] → 256-768 semantic dimensions
    ↓
[Combined Vector] → 290-802 total features
    ↓
[Random Forest] → Classification + Confidence
```

### Training Pipeline
1. Load datasets from multiple directories
2. Train LSA encoder on email corpus (TF-IDF + SVD)
3. Extract combined features (numeric + semantic)
4. Train Random Forest with cross-entropy loss
5. Evaluate on held-out test set
6. Save models for deployment

## Security & Privacy

### Current Implementation
- **In-Memory Processing**: Emails processed without persistence
- **No Data Collection**: No raw emails stored
- **Metadata Only**: URL analysis via HEAD requests (no active fetching)
- **Local Inference**: All processing happens locally

### Planned Security Features
- **Content Safety**: Never auto-fetch attachments
- **Rate Limiting**: Prevention of adversarial usage
- **PII Redaction**: Anonymization of stored samples
- **OAuth Security**: Least-privilege scopes for email access

## Dataset Information

### Included Datasets
- **easy_ham**: 2,551 legitimate emails
- **spam**: 501 spam emails

### Additional Available Datasets
Compressed archives in `Datasets/` (extract to add more training data):
- `spam_2.tar.bz2`, `spam_3.tar.bz2`, `spam_4.tar.bz2`
- `easy_ham_2.tar.bz2`, `easy_ham_3.tar.bz2`
- `hard_ham.tar.bz2`, `hard_ham_2.tar.bz2`

Extract with:
```bash
cd Datasets
tar -xjf spam_2.tar.bz2
tar -xjf easy_ham_2.tar.bz2
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run preprocessing tests:
```bash
python -m pytest tests/test_preprocessing.py -v
```



