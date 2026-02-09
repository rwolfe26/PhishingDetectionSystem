# Complete Email Phishing Detection Pipeline

This is the complete, production-ready pipeline that combines preprocessing and LSA semantic analysis for email phishing detection.

## What It Does

The pipeline integrates:
1. **Email Preprocessing** (34 numeric features)
   - URL analysis (IP addresses, HTTPS, suspicious ports, etc.)
   - Text analysis (word counts, special characters, etc.)
   - Header analysis (reply-to mismatches, X-Mailer, etc.)
   - HTML analysis (forms, iframes, hidden text, etc.)
   - Attachment analysis (executables, archives, etc.)

2. **LSA Semantic Analysis** (up to 768 dimensions)
   - TF-IDF vectorization with n-grams
   - Truncated SVD for dimensionality reduction
   - L2 normalization for stability

3. **Random Forest Classifier**
   - Trained on combined feature vectors
   - Provides probability scores for risk assessment

## Results

Training on 3,052 emails (2,551 ham + 501 spam):
- **Accuracy: 99.35%**
- **Precision: 98.98%**
- **Recall: 97.00%**
- **F1-Score: 97.98%**

Error Analysis:
- False Positives: 1 (0.20% of ham emails)
- False Negatives: 3 (3.00% of spam emails)

## Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Dependencies are already installed:
# - scikit-learn
# - joblib
# - numpy
```

## Usage

### 1. Train the Pipeline

```bash
python pipeline.py --train
```

This will:
- Load ham and spam emails from `Datasets/` directory
- Split into 80% training, 20% test
- Fit LSA encoder on training data
- Extract combined features (preprocessing + LSA)
- Train Random Forest classifier
- Evaluate on test set
- Save models to `models/` directory

**Training Options:**
```bash
python pipeline.py --train \
  --lsa-components 256 \      # Number of LSA dimensions (default: 768)
  --lsa-min-df 2 \           # Minimum document frequency (default: 2)
  --lsa-max-df 0.85 \        # Maximum document frequency (default: 0.85)
  --test-size 0.2            # Test set proportion (default: 0.2)
```

### 2. Predict Single Emails

```bash
python pipeline.py --predict path/to/email.txt
```

Example:
```bash
# Test on a spam email
python pipeline.py --predict Datasets/spam/0000.7b1b73cf36cf9dbc3d64e3f2ee2b91f1

# Test on a legitimate email
python pipeline.py --predict Datasets/easy_ham/0001.ea7e79d3153e7469e7a9c3e0af6a357e
```

Output:
```
============================================================
Prediction Result
============================================================
Email: Datasets/spam/0000.7b1b73cf36cf9dbc3d64e3f2ee2b91f1
Prediction: SPAM
Confidence: 62.85%
Risk Level: LOW (Possibly spam)
```

Risk Levels:
- **HIGH**: >90% confidence spam
- **MEDIUM**: 70-90% confidence spam
- **LOW**: 50-70% confidence spam
- **Safe**: <50% confidence spam (legitimate)

## Architecture

### File Structure

```
Detection_System/
├── pipeline.py                           # Complete pipeline (this file)
├── preprocessing/                        # Preprocessing module
│   ├── __init__.py                      # Integration functions
│   ├── email_parser.py                  # RFC 822 email parsing
│   ├── url_extractor.py                 # URL analysis
│   └── feature_extractor.py             # 34 numeric features
├── bert_base/
│   └── lsa_tool.py                      # LSA semantic analysis
├── models/                              # Saved models (created after training)
│   ├── lsa_encoder.pkl                  # Trained LSA encoder
│   ├── phishing_classifier.pkl          # Trained classifier
│   └── pipeline_metadata.pkl            # Pipeline configuration
└── Datasets/
    ├── easy_ham/                        # Ham (legitimate) emails
    └── spam/                            # Spam emails
```

### Pipeline Class

The `EmailPhishingPipeline` class provides:

```python
pipeline = EmailPhishingPipeline(
    lsa_components=768,
    lsa_min_df=2,
    lsa_max_df=0.85
)

# Training workflow
emails, labels = pipeline.load_dataset(ham_dirs, spam_dirs)
pipeline.fit_lsa(training_emails)
X = pipeline.extract_features(emails)
pipeline.train_classifier(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)
pipeline.save_models(output_dir)

# Prediction workflow
pipeline.load_models(model_dir)
prediction, probability = pipeline.predict_single(email_text)
```

## Feature Importance

Top features identified by the model:

1. **Feature 23** (numeric): Has hidden text - 9.54%
2. **Feature 27** (numeric): Number of external links - 6.54%
3. **LSA dimension 7**: Semantic context - 5.33%
4. **Feature 32** (numeric): Has archive attachment - 5.27%
5. **Feature 16** (numeric): Subject has "Re:" or "Fw:" - 4.94%

LSA dimensions capture semantic patterns that distinguish phishing attempts from legitimate emails.

## Saved Models

After training, three files are saved:

1. **lsa_encoder.pkl**: Fitted LSA encoder
   - TF-IDF vectorizer with vocabulary
   - SVD transformation matrix
   - Can be reused on new emails

2. **phishing_classifier.pkl**: Trained Random Forest
   - 100 decision trees
   - Feature importances
   - Ready for prediction

3. **pipeline_metadata.pkl**: Configuration
   - Number of LSA components
   - Feature dimensions
   - Min/max document frequency settings

## Using in Your Own Code

```python
from pathlib import Path
import joblib
from preprocessing import preprocess_email_with_lsa

# Load the saved models
model_dir = Path('models')
lsa_encoder = joblib.load(model_dir / 'lsa_encoder.pkl')
classifier = joblib.load(model_dir / 'phishing_classifier.pkl')

# Process a new email
raw_email = open('new_email.txt').read()
result = preprocess_email_with_lsa(raw_email, lsa_encoder)

# Get prediction
X = result['combined_vector'].reshape(1, -1)
prediction = classifier.predict(X)[0]
probability = classifier.predict_proba(X)[0][1]

print(f"Spam probability: {probability:.2%}")
```

## Extending the Pipeline

### Add More Data

To improve accuracy, add more spam/ham directories to training:

```python
# Edit pipeline.py, line ~390
ham_dirs = [
    dataset_dir / 'easy_ham',
    dataset_dir / 'easy_ham_2',    # Add this
    dataset_dir / 'hard_ham',      # Add this
]
spam_dirs = [
    dataset_dir / 'spam',
    dataset_dir / 'spam_2',        # Add this
]
```

### Try Different Classifiers

Replace Random Forest with other algorithms:

```python
# In train_classifier() method
from sklearn.ensemble import GradientBoostingClassifier
self.classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Or try XGBoost
import xgboost as xgb
self.classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)

# Or neural network
from sklearn.neural_network import MLPClassifier
self.classifier = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42)
```

### Hyperparameter Tuning

Use GridSearchCV for better parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

self.classifier = grid_search.best_estimator_
```

## Performance Tips

1. **Training Time**: ~2-3 minutes for 3,000 emails
2. **Prediction Time**: <100ms per email
3. **Memory Usage**: ~100MB for model + encoder
4. **Batch Processing**: Process multiple emails at once for better throughput

## Troubleshooting

### "No spam emails found"
Extract the spam archives:
```bash
cd Datasets
tar -xjf spam.tar.bz2
tar -xjf spam_2.tar.bz2
```

### "Not enough LSA components"
Reduce `--lsa-components` based on your corpus size:
```bash
python pipeline.py --train --lsa-components 128
```

### Low accuracy
- Add more training data
- Try different `lsa_min_df` and `lsa_max_df` values
- Experiment with classifier parameters
- Extract additional spam archives

## Next Steps

1. Extract additional datasets (spam_2, spam_3, spam_4, hard_ham, etc.)
2. Implement cross-validation
3. Add ensemble methods (voting classifier)
4. Create web API for production deployment
5. Add real-time monitoring and retraining

## Related Files

- `example_lsa_preprocessing_integration.py`: Simple integration example
- `LSA_PREPROCESSING_INTEGRATION.md`: Detailed integration documentation
- `tests/test_preprocessing.py`: Unit tests for preprocessing
- `bert_base/lsa_research_report.md`: LSA methodology documentation
