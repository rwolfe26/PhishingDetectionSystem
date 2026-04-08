# Refactoring Summary

This document summarises the evolution of the Email Phishing Detection System from a monolithic script to a modular, extensible pipeline.

## Phase 1 — Initial Prototype

**File:** `old_models/pipeline_monolithic_backup.py`

The original implementation was a single ~510-line Python file containing the entire pipeline:
- Email loading
- Feature extraction
- LSA fitting
- Random Forest training
- CLI interface

This worked for initial experiments but was difficult to test, extend, or reuse.

## Phase 2 — Preprocessing Module

**Directory:** `preprocessing/`

The email processing logic was extracted into a dedicated package:

- `email_parser.py` — RFC 822 email parsing (`ParsedEmail`, `AttachmentInfo`)
- `url_extractor.py` — URL extraction and analysis (`URLExtractor`, `URLInfo`)
- `feature_extractor.py` — 34 numeric features (`EmailFeatures`, `FeatureExtractor`)
- `__init__.py` — Public API that wires parsing → URL extraction → features → LSA embedding

## Phase 3 — LSA Semantic Analysis

**Directory:** `bert_base/`

A TF-IDF + TruncatedSVD + L2-normalisation stack was added to produce semantic embeddings alongside the 34 numeric features. The directory is named `bert_base` for historical reasons (BERT was considered but sklearn LSA was chosen for speed and no-GPU requirement).

- `lsa_tool.py` — `LSATextEncoder` wrapping sklearn's TF-IDF vectoriser and TruncatedSVD

The preprocessing `__init__.py` exposes `fit_lsa_encoder` and `preprocess_email_with_lsa` to combine both feature types into a single vector (34 numeric + N LSA dimensions).

## Phase 4 — Modular Pipeline Package

**Directory:** `pipeline/`

The monolithic pipeline was split into focused modules:

| Module | Responsibility |
|--------|---------------|
| `core.py` | `EmailPhishingPipeline` — orchestrates LSA fitting, feature extraction, model save/load |
| `data_loader.py` | `DataLoader` — loads raw email files from one or more directories |
| `trainer.py` | `Trainer` — trains and evaluates the Random Forest classifier |
| `predictor.py` | `Predictor` — single-email inference and result formatting |

**Entry point:** `run_pipeline.py` replaced the old `pipeline.py` CLI.

## Key Design Decisions

- **LSA over BERT**: Training and inference run on CPU-only machines; LSA achieves good semantic compression with no GPU requirement.
- **34 handcrafted numeric features**: Interpretable, fast to compute, and complementary to LSA embeddings.
- **Random Forest**: Good out-of-the-box performance, native feature importance, and resistance to overfitting with the default `max_depth=20` constraint.
- **Stratified train/test split**: Preserves the ham/spam ratio across splits given the class imbalance (≈5:1).

## Current Limitations and Next Steps

See `README.md` roadmap for planned improvements including:
- Cross-validation and error analysis
- Model benchmarking (XGBoost, SVM, ensembles)
- Additional phishing-specific features
- Web interface with explainability
