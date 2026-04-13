# LSA + Preprocessing Integration

This document explains how the LSA tool has been integrated with the preprocessing pipeline for the email phishing detection system.

## Overview

The integration connects two key components:
1. **Preprocessing Pipeline**: Extracts 34 numeric features from emails (URLs, text patterns, headers, HTML, attachments)
2. **LSA Tool**: Generates semantic embeddings (up to 768 dimensions) from email text

The combined output provides rich feature vectors for machine learning models.

## Architecture

```
Raw Email Text
    ↓
[Email Parser] → ParsedEmail (subject, body, headers, attachments)
    ↓
[URL Extractor] → List[URLInfo] (domains, paths, suspicious patterns)
    ↓
[Feature Extractor] → 34 numeric features
    ↓
[LSA Encoder] → N-dimensional semantic embedding (e.g., 768 dims)
    ↓
Combined Feature Vector: [34 numeric features] + [N LSA dimensions]
```

## Installation

Install required dependencies:

```bash
source .venv/bin/activate
pip install scikit-learn joblib numpy
```

## Usage

### 1. Fit the LSA Encoder

First, train the LSA encoder on your corpus of emails:

```python
from preprocessing import fit_lsa_encoder

# Load your training emails (raw RFC 822 format)
training_emails = load_your_training_data()

# Fit the encoder
lsa_encoder = fit_lsa_encoder(
    training_emails,
    n_components=768,    # Target dimensions (may be reduced automatically)
    min_df=2,            # Ignore very rare terms
    max_df=0.85          # Ignore very common terms
)

# Save the encoder for later use
import joblib
joblib.dump(lsa_encoder, 'lsa_encoder.pkl')
```

**Important**: The actual number of LSA components depends on your corpus size:
- With 500 emails, you might get ~500 components
- With 2,551 emails (full dataset), you can get closer to 768
- The encoder automatically reduces dimensions if needed

### 2. Process Individual Emails

Use the integrated pipeline to process single emails:

```python
from preprocessing import preprocess_email_with_lsa

result = preprocess_email_with_lsa(raw_email, lsa_encoder)

# Access the results
print(result['subject'])              # Email subject
print(result['from_address'])         # Sender address
print(result['urls'])                 # List of URLInfo objects
print(result['feature_dict'])         # Dict of 34 numeric features
print(result['lsa_embedding'])        # Array of shape (N,) - LSA embedding
print(result['combined_vector'])      # Array of shape (34+N,) - Full feature vector
```

The result dictionary contains:
- `subject`: Email subject string
- `body_text`: Plain text body
- `from_address`: Sender email
- `urls`: List of URLInfo objects with analysis
- `features`: EmailFeatures object with 34 numeric features
- `feature_vector`: List of 34 numeric values
- `lsa_embedding`: numpy array (N,) with semantic embedding
- `combined_vector`: numpy array (34+N,) ready for ML models

### 3. Batch Processing

Process multiple emails efficiently:

```python
from preprocessing import preprocess_email_batch_with_lsa
import numpy as np

results = preprocess_email_batch_with_lsa(test_emails, lsa_encoder)

# Extract feature matrix for ML model
X = np.array([r['combined_vector'] for r in results])
# X.shape = (num_emails, 34 + N_components)

# Extract labels if you have them
y = np.array([r['label'] for r in your_labeled_data])

# Train your classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
```

### 4. Convenience Function for Quick Extraction

For streamlined feature extraction:

```python
from preprocessing import get_combined_feature_vector

# Get just the feature vector
vector = get_combined_feature_vector(raw_email, lsa_encoder)
# vector.shape = (34 + N_components,)

# Build feature matrix for multiple emails
X = np.array([
    get_combined_feature_vector(email, lsa_encoder)
    for email in emails
])
```

## Feature Breakdown

### Numeric Features (34 dimensions)

The first 34 dimensions contain:

| Category | Count | Examples |
|----------|-------|----------|
| URL-based | 10 | num_urls, has_ip_url, no_https_ratio, suspicious_port |
| Text-based | 4 | num_words, num_chars, num_special_chars |
| Keyword-based | 3 | urgent_keywords, credential_keywords, action_keywords |
| Header-based | 6 | reply_to_mismatch, suspicious_mailer, num_hops |
| HTML-based | 5 | has_form, has_iframe, has_hidden_text |
| Attachment-based | 3 | num_attachments, has_executable, has_archive |

### LSA Semantic Embedding (up to 768 dimensions)

The remaining dimensions contain semantic information from:
- Email subject
- Email body text
- Preprocessed with TF-IDF and Truncated SVD
- L2-normalized for stability

The actual number of LSA dimensions depends on your corpus size and vocabulary.

## Example: Complete Workflow

```python
import numpy as np
from pathlib import Path
import joblib
from preprocessing import (
    fit_lsa_encoder,
    preprocess_email_batch_with_lsa
)

# 1. Load training data
def load_emails(directory):
    emails = []
    for file in Path(directory).glob('*'):
        if file.is_file():
            emails.append(file.read_text(encoding='utf-8', errors='ignore'))
    return emails

ham_emails = load_emails('Datasets/easy_ham')
spam_emails = load_emails('Datasets/spam')
all_training = ham_emails + spam_emails

# 2. Fit LSA encoder
encoder = fit_lsa_encoder(all_training, n_components=768, min_df=2)
joblib.dump(encoder, 'models/lsa_encoder.pkl')

# 3. Create labeled dataset
X_ham_results = preprocess_email_batch_with_lsa(ham_emails, encoder)
X_spam_results = preprocess_email_batch_with_lsa(spam_emails, encoder)

X_ham = np.array([r['combined_vector'] for r in X_ham_results])
X_spam = np.array([r['combined_vector'] for r in X_spam_results])

X = np.vstack([X_ham, X_spam])
y = np.array([0] * len(X_ham) + [1] * len(X_spam))  # 0=ham, 1=spam

# 4. Train classifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Save model
joblib.dump(clf, 'models/phishing_classifier.pkl')
```

## Advanced Usage

### Reusing a Saved Encoder

```python
import joblib
from preprocessing import preprocess_email_with_lsa

# Load the encoder
encoder = joblib.load('lsa_encoder.pkl')

# Process new emails
result = preprocess_email_with_lsa(new_email, encoder)
```

### Feature Importance Analysis

```python
from preprocessing import EmailFeatures

# Get feature names
numeric_feature_names = EmailFeatures.feature_names()
lsa_feature_names = [f'lsa_{i}' for i in range(lsa_encoder.n_components)]
all_feature_names = numeric_feature_names + lsa_feature_names

# After training a model, check feature importance
if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
    for name, imp in sorted(zip(all_feature_names, importance),
                           key=lambda x: x[1], reverse=True)[:20]:
        print(f"{name}: {imp:.4f}")
```

### Customizing LSA Parameters

```python
# For smaller datasets, use fewer components
encoder = fit_lsa_encoder(
    emails,
    n_components=256,      # Fewer dimensions
    max_features=5000,     # Limit vocabulary size
    min_df=3,              # Require term appears in 3+ docs
    max_df=0.7             # Ignore very common terms
)

# For larger datasets, you can increase
encoder = fit_lsa_encoder(
    emails,
    n_components=768,      # More dimensions
    max_features=None,     # No limit on vocabulary
    min_df=2,              # Include rarer terms
    max_df=0.95            # Allow more common terms
)
```

## Files Modified

- `preprocessing/__init__.py`: Added 4 new functions:
  - `fit_lsa_encoder()`: Train LSA on email corpus
  - `preprocess_email_with_lsa()`: Single email with LSA
  - `preprocess_email_batch_with_lsa()`: Batch processing with LSA
  - `get_combined_feature_vector()`: Convenience function

## Performance Notes

- **LSA Fitting**: O(n_emails × vocabulary_size) - can take 1-2 minutes for 2,500 emails
- **Transform**: Very fast (<1ms per email) after fitting
- **Memory**: Encoder size ~10-50MB depending on vocabulary
- **Parallelization**: Batch processing is sequential but can be parallelized if needed

## Troubleshooting

### "n_components too large"

If you get an error about n_components being too large:
- Your corpus is too small for the requested dimensions
- Either reduce `n_components` or increase your training corpus size
- The encoder will automatically adjust, but you can set it explicitly

### Different dimensions than expected

The actual LSA dimensions depend on:
- Number of unique terms in your corpus (vocabulary size)
- The `min_df` and `max_df` filters you apply
- The number of training documents

For 768 dimensions, you typically need at least 1,000+ documents with good vocabulary diversity.

## Next Steps

1. Train on your full dataset (easy_ham + spam + hard_ham)
2. Experiment with different classifiers:
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Logistic Regression
   - Neural Networks
3. Perform hyperparameter tuning
4. Add cross-validation
5. Deploy the model with the saved encoder

## References

- Original LSA research: `bert_base/lsa_research_report.md`
- Preprocessing design: `preprocessing/` module
- Email parser: `preprocessing/email_parser.py`
- Feature extractor: `preprocessing/feature_extractor.py`
