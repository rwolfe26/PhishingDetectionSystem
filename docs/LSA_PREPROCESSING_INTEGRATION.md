# LSA + Preprocessing Integration

This document explains how the LSA semantic analysis is integrated with the preprocessing pipeline.

## Overview

The integration connects two components:

1. **Preprocessing Pipeline** — extracts 42 numeric features from emails (URLs, text, headers, HTML, attachments, and phishing-specific signals)
2. **LSA Tool** — generates semantic embeddings (configurable N dimensions, default 128) from email text

The combined output provides rich feature vectors for the classifier.

## Architecture

```
Raw Email Text
    ↓
[Email Parser] → ParsedEmail (subject, body, headers, attachments)
    ↓
[URL Extractor] → List[URLInfo] (domains, paths, suspicious patterns)
    ↓
[Feature Extractor] → 42 numeric features
    ↓
[LSA Encoder] → N-dimensional semantic embedding (e.g., 128 dims)
    ↓
Combined Feature Vector: [42 numeric] + [N LSA dims] = (42+N,)
```

## Efficient Single-Pass Training

The key efficiency improvement is `fit_lsa_and_extract()` in `EmailPhishingPipeline`, which
parses each email exactly once during training:

```python
# OLD approach (2× slower — emails parsed twice)
pipeline.fit_lsa(train_emails)          # parse → extract text → fit LSA
X_train = pipeline.extract_features(train_emails)  # parse → extract features → LSA transform

# NEW approach (single pass)
X_train = pipeline.fit_lsa_and_extract(train_emails)
# Internally: parse → collect body_texts + numeric features → fit LSA → batch transform
```

For large datasets (10k+ emails), this halves the preprocessing time.

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Fit the LSA Encoder

```python
from preprocessing import fit_lsa_encoder

# From raw emails (also preprocesses to extract body text)
lsa = fit_lsa_encoder(training_emails, n_components=128, min_df=2, max_df=0.85)

# From pre-extracted body texts (faster — skip email parsing)
from preprocessing import fit_lsa_encoder_from_texts
lsa = fit_lsa_encoder_from_texts(body_texts, n_components=128)

import joblib
joblib.dump(lsa, 'models/lsa_encoder.pkl')
```

### Process a Single Email

```python
from preprocessing import preprocess_email_with_lsa
import joblib

lsa = joblib.load('models/lsa_encoder.pkl')
result = preprocess_email_with_lsa(raw_email, lsa)

print(result['lsa_embedding'].shape)    # (128,)
print(result['combined_vector'].shape)  # (170,)  → 42 numeric + 128 LSA
```

### Batch Processing

```python
from preprocessing import preprocess_email_batch_with_lsa
import numpy as np

results = preprocess_email_batch_with_lsa(email_list, lsa)
X = np.array([r['combined_vector'] for r in results])  # (n_emails, 42+N)
```

### Single-Pass Fit + Extract (Most Efficient)

```python
from preprocessing import fit_and_extract_features

# Returns (feature_matrix, body_texts) in one parse pass
X, body_texts = fit_and_extract_features(train_emails, lsa)
```

## The 42 Numeric Features

| Category | Count | Examples |
|----------|-------|---------|
| URL | 10 | `num_urls`, `has_ip_url`, `no_https_ratio`, `num_shortener_urls`, `brand_impersonation_score` |
| Text | 4 | `num_words`, `num_chars`, `num_special_chars` |
| Keywords | 3 | `num_urgent_keywords`, `num_credential_keywords`, `num_action_keywords` |
| Headers | 6 | `has_reply_to_mismatch`, `spf_dkim_fail`, `sender_domain_mismatch` |
| HTML | 6 | `has_form`, `has_iframe`, `has_hidden_text` |
| Attachments | 3 | `has_executable_attachment`, `has_archive_attachment` |
| Phishing-specific | 10 | `num_homograph_chars`, `urgency_density`, `greeting_generic`, `subject_all_caps_ratio` |

Get the ordered list of feature names at runtime:

```python
from preprocessing import EmailFeatures
print(EmailFeatures.feature_names())  # list of 42 names
```

## LSA Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_components` | 128 | Dimensions of semantic embedding. Start with 128; run `experiments/lsa_dimension_search.py` to find the optimal value. |
| `min_df` | 2 | Terms must appear in ≥ 2 documents. Raise to 5+ for large corpora. |
| `max_df` | 0.85 | Terms in > 85% of documents are dropped (too common). |
| `max_features` | None | Optional vocabulary cap. Set to 50000 for memory-constrained environments. |

## LSA Dimensionality Search

Run `experiments/lsa_dimension_search.py` to measure accuracy vs. component count:

```bash
python experiments/lsa_dimension_search.py --max-emails 3000
```

Results are saved to `experiments/lsa_dimension_results.csv`. Typically values in the
range 100–256 give the best F1 with this dataset.

## Example: Full Training + Prediction

```python
from pathlib import Path
from pipeline import EmailPhishingPipeline, DataLoader, Trainer, Predictor

# Load data
loader = DataLoader()
emails, labels = loader.load_dataset(
    ham_dirs=[Path('Datasets/easy_ham'), Path('Datasets/easy_ham_2')],
    spam_dirs=[Path('Datasets/spam'), Path('Datasets/spam_2')],
)
csv_emails, csv_labels = loader.load_phishing_csv(
    Path('Datasets/Phishing_Email.csv'), max_samples=2000
)
import numpy as np
all_emails = emails + csv_emails
all_labels = np.concatenate([labels, csv_labels])

# Train/test split
from sklearn.model_selection import train_test_split
X_tr_emails, X_te_emails, y_tr, y_te = train_test_split(
    all_emails, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# Single-pass: parse once, fit LSA, extract features
pipeline = EmailPhishingPipeline(lsa_components=128)
X_train = pipeline.fit_lsa_and_extract(X_tr_emails)
X_test = pipeline.extract_features(X_te_emails)

# Train and evaluate
trainer = Trainer()
trainer.train_classifier(pipeline, X_train, y_tr)
trainer.evaluate(pipeline, X_test, y_te)

# Save
pipeline.save_models(Path('models'))

# Later: load and predict
pipeline.load_models(Path('models'))
pred, prob = Predictor().predict_single(pipeline, open('email.txt').read())
print(f"{'PHISHING' if pred == 1 else 'HAM'}  —  confidence: {prob:.1%}")
```

## Troubleshooting

### ValueError: After pruning, no terms remain
Occurs when `min_df` is too high for a small corpus. Use `min_df=1` with
fewer than ~100 training emails.

### Combined vector has wrong shape
If the LSA encoder was trained with `n_components=768` but you changed it to `128`,
the saved models will be incompatible. Retrain from scratch after changing `lsa_components`.

### Memory errors on large corpora
- Set `max_features=50000` in `fit_lsa_encoder` to cap vocabulary size
- Reduce `n_components` to 64 or 128
- Process in batches using `preprocess_email_batch_with_lsa`
