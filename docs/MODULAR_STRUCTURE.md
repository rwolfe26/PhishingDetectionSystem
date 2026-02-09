# Modular Pipeline Structure

The pipeline has been reorganized into a clean, modular structure for easier maintenance and development.

## Directory Structure

```
Detection_System/
├── run_pipeline.py              # CLI entry point
├── pipeline/                    # Pipeline package
│   ├── __init__.py             # Package initialization
│   ├── core.py                 # EmailPhishingPipeline class
│   ├── data_loader.py          # Data loading utilities
│   ├── trainer.py              # Training and evaluation
│   └── predictor.py            # Prediction logic
├── preprocessing/               # Preprocessing module
│   ├── __init__.py             # Integration with LSA
│   ├── email_parser.py         # Email parsing
│   ├── url_extractor.py        # URL extraction
│   └── feature_extractor.py   # Feature extraction
├── bert_base/
│   └── lsa_tool.py             # LSA semantic analysis
└── models/                      # Saved models
    ├── lsa_encoder.pkl
    ├── phishing_classifier.pkl
    └── pipeline_metadata.pkl
```

## Module Descriptions

### `run_pipeline.py`
**Main CLI entry point**
- Handles command-line arguments
- Orchestrates training and prediction workflows
- Clean interface for users

Functions:
- `train_pipeline(args)` - Complete training workflow
- `predict_email(args)` - Single email prediction
- `main()` - CLI argument parsing

### `pipeline/core.py`
**Core pipeline class**
- `EmailPhishingPipeline` - Main pipeline orchestrator
- Manages LSA encoder and classifier
- Handles model saving/loading

Key Methods:
- `fit_lsa(emails)` - Train LSA encoder
- `extract_features(emails)` - Generate feature vectors
- `save_models(output_dir)` - Persist models to disk
- `load_models(model_dir)` - Load trained models

### `pipeline/data_loader.py`
**Data loading utilities**
- `DataLoader` - Static utility class
- Loads emails from directories
- Organizes datasets with labels

Key Methods:
- `load_emails_from_directory(directory, label)` - Load single directory
- `load_dataset(ham_dirs, spam_dirs)` - Load complete dataset

### `pipeline/trainer.py`
**Training and evaluation**
- `Trainer` - Static training utility class
- Trains Random Forest classifier
- Evaluates model performance

Key Methods:
- `train_classifier(pipeline, X_train, y_train)` - Train model
- `evaluate(pipeline, X_test, y_test)` - Evaluate and report metrics

### `pipeline/predictor.py`
**Prediction logic**
- `Predictor` - Static prediction utility class
- Classifies new emails
- Formats prediction results

Key Methods:
- `predict_single(pipeline, email)` - Predict one email
- `format_prediction_result(email_path, prediction, probability)` - Display result

## Usage

### Training
```bash
# Basic training
python run_pipeline.py --train

# With custom parameters
python run_pipeline.py --train \
  --lsa-components 256 \
  --lsa-min-df 2 \
  --lsa-max-df 0.85 \
  --test-size 0.2
```

### Prediction
```bash
# Predict a single email
python run_pipeline.py --predict path/to/email.txt

# Examples
python run_pipeline.py --predict Datasets/spam/0000.7b1b73cf36cf9dbc3d64e3f2ee2b91f1
python run_pipeline.py --predict Datasets/easy_ham/0001.ea7e79d3153e7469e7a9c3e0af6a357e
```

### Help
```bash
python run_pipeline.py --help
```

## Using the Pipeline Programmatically

The modular structure makes it easy to use in your own code:

```python
from pathlib import Path
from pipeline import EmailPhishingPipeline, DataLoader, Trainer, Predictor

# Initialize pipeline
pipeline = EmailPhishingPipeline(lsa_components=256)

# Load data
data_loader = DataLoader()
emails, labels = data_loader.load_dataset(ham_dirs, spam_dirs)

# Train
pipeline.fit_lsa(emails)
X = pipeline.extract_features(emails)

trainer = Trainer()
trainer.train_classifier(pipeline, X_train, y_train)
trainer.evaluate(pipeline, X_test, y_test)

# Save
pipeline.save_models(Path('models'))

# Later: Load and predict
pipeline.load_models(Path('models'))
predictor = Predictor()
prediction, probability = predictor.predict_single(pipeline, email_text)
```

## Benefits of Modular Structure

1. **Separation of Concerns**
   - Each module has a single, clear responsibility
   - Easier to understand and maintain

2. **Testability**
   - Each module can be tested independently
   - Easy to write unit tests for specific components

3. **Reusability**
   - Import only what you need
   - Use components in other projects

4. **Extensibility**
   - Easy to add new features without touching existing code
   - Replace components (e.g., swap classifier) easily

5. **Readability**
   - Smaller files are easier to navigate
   - Clear naming makes intent obvious

## Adding New Features

### Add a New Classifier
Edit `pipeline/trainer.py`:
```python
from sklearn.ensemble import GradientBoostingClassifier

@staticmethod
def train_classifier(pipeline, X_train, y_train):
    pipeline.classifier = GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
    pipeline.classifier.fit(X_train, y_train)
```

### Add Cross-Validation
Create `pipeline/cross_validator.py`:
```python
from sklearn.model_selection import cross_val_score

class CrossValidator:
    @staticmethod
    def validate(pipeline, X, y, cv=5):
        scores = cross_val_score(
            pipeline.classifier, X, y, cv=cv
        )
        return scores
```

### Add Batch Prediction
Edit `pipeline/predictor.py`:
```python
@staticmethod
def predict_batch(pipeline, emails):
    results = []
    for email in emails:
        pred, prob = Predictor.predict_single(pipeline, email)
        results.append((pred, prob))
    return results
```

## File Comparison

**Before:** Single monolithic `pipeline.py` (510 lines)

**After:** Modular structure
- `run_pipeline.py` (146 lines) - CLI
- `pipeline/core.py` (167 lines) - Core logic
- `pipeline/data_loader.py` (80 lines) - Data loading
- `pipeline/trainer.py` (110 lines) - Training
- `pipeline/predictor.py` (49 lines) - Prediction

Total: ~552 lines, but much more organized and maintainable!

## Migration from Old Pipeline

The old `pipeline.py` is still available for reference, but use `run_pipeline.py` instead:

```bash
# Old way (still works)
python pipeline.py --train

# New way (recommended)
python run_pipeline.py --train
```

All functionality is preserved, just better organized!
