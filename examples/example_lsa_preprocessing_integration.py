"""
Example: Using the integrated LSA + Preprocessing Pipeline

This script demonstrates how to connect the LSA tool with preprocessing
to create a complete feature extraction pipeline for phishing detection.

The pipeline produces 802-dimensional feature vectors:
- First 34 dimensions: Numeric features (URLs, text, headers, HTML, attachments)
- Last 768 dimensions: Semantic embeddings from LSA
"""

import os
from pathlib import Path
import numpy as np

# Import the integrated preprocessing functions
from preprocessing import (
    fit_lsa_encoder,
    preprocess_email_with_lsa,
    preprocess_email_batch_with_lsa,
    get_combined_feature_vector
)


def load_emails_from_directory(directory: Path, limit: int = None):
    """Load raw email files from a directory."""
    emails = []
    for i, filename in enumerate(os.listdir(directory)):
        if limit and i >= limit:
            break
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                emails.append(f.read())
    return emails


def main():
    # Define paths (resolve to repo root, not examples/)
    base_dir = Path(__file__).resolve().parent.parent
    ham_dir = base_dir / 'Datasets' / 'easy_ham'
    spam_dir = base_dir / 'Datasets' / 'spam'

    print("=" * 60)
    print("LSA + Preprocessing Integration Example")
    print("=" * 60)

    # Step 1: Load training emails
    print("\n[Step 1] Loading training emails...")
    training_emails = load_emails_from_directory(ham_dir, limit=500)
    print(f"Loaded {len(training_emails)} training emails")

    # Step 2: Fit LSA encoder on the training corpus
    print("\n[Step 2] Fitting LSA encoder on training corpus...")
    lsa_encoder = fit_lsa_encoder(
        training_emails,
        n_components=768,      # Match BERT-like dimensions
        min_df=2,              # Ignore very rare terms
        max_df=0.85            # Ignore very common terms
    )
    print("LSA encoder fitted successfully!")

    # Step 3: Process a single test email with the complete pipeline
    print("\n[Step 3] Processing test email with integrated pipeline...")
    test_email = training_emails[0]  # Use first email as example

    result = preprocess_email_with_lsa(test_email, lsa_encoder)

    print(f"\nPreprocessing results:")
    print(f"  - Subject: {result['subject'][:50]}...")
    print(f"  - From: {result['from_address']}")
    print(f"  - URLs found: {len(result['urls'])}")
    print(f"  - Attachments: {len(result['attachments'])}")
    print(f"\nFeature extraction:")
    print(f"  - Numeric features (34 dims): {result['feature_vector'][:5]}... (showing first 5)")
    print(f"  - LSA embedding (768 dims): {result['lsa_embedding'][:5]}... (showing first 5)")
    print(f"  - Combined vector shape: {result['combined_vector'].shape}")

    # Step 4: Batch processing for ML model training
    print("\n[Step 4] Batch processing multiple emails...")
    test_emails = training_emails[100:110]  # Process emails 100-110
    batch_results = preprocess_email_batch_with_lsa(test_emails, lsa_encoder)

    # Extract combined feature matrix for ML model
    X = np.array([r['combined_vector'] for r in batch_results])
    print(f"Feature matrix shape: {X.shape}")
    print(f"  - Shape: (num_emails={X.shape[0]}, features={X.shape[1]})")
    print(f"  - Features breakdown: 34 numeric + 768 LSA = 802 total")

    # Step 5: Quick extraction function (convenience wrapper)
    print("\n[Step 5] Using convenience function...")
    single_vector = get_combined_feature_vector(test_emails[0], lsa_encoder)
    print(f"Single vector shape: {single_vector.shape}")

    # Step 6: Show how to save the encoder for later use
    print("\n[Step 6] Saving the LSA encoder...")
    try:
        import joblib
        encoder_path = base_dir / 'models' / 'lsa_encoder_768d.pkl'
        joblib.dump(lsa_encoder, encoder_path)
        print(f"Encoder saved to: {encoder_path}")
        print("\nTo load later:")
        print("  import joblib")
        print(f"  encoder = joblib.load('{encoder_path}')")
    except ImportError:
        print("Install joblib to save the encoder: pip install joblib")

    print("\n" + "=" * 60)
    print("Integration complete! You can now:")
    print("  1. Train a classifier on the 802-dimensional feature vectors")
    print("  2. Use the fitted encoder for consistent embeddings")
    print("  3. Combine both preprocessing and LSA in one pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
