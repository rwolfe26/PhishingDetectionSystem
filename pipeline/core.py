"""
Core Pipeline Class

Main EmailPhishingPipeline class that orchestrates the complete workflow.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import joblib

from preprocessing import (
    fit_lsa_encoder,
    preprocess_email_batch_with_lsa,
    preprocess_email_with_lsa,
)


class EmailPhishingPipeline:
    """
    Complete pipeline for email phishing detection.

    Combines preprocessing (34 numeric features) with LSA semantic analysis
    (up to 768 dimensions) to create rich feature vectors for classification.
    """

    def __init__(self,
                 lsa_components: int = 768,
                 lsa_min_df: int = 2,
                 lsa_max_df: float = 0.85):
        """
        Initialize the pipeline.

        Args:
            lsa_components: Number of LSA dimensions
            lsa_min_df: Minimum document frequency for LSA
            lsa_max_df: Maximum document frequency for LSA
        """
        self.lsa_components = lsa_components
        self.lsa_min_df = lsa_min_df
        self.lsa_max_df = lsa_max_df

        self.lsa_encoder = None
        self.classifier = None
        self.feature_dim = None

    def fit_lsa(self, emails: List[str]):
        """
        Fit the LSA encoder on training emails.

        Args:
            emails: List of raw email texts
        """
        print(f"\n{'='*60}")
        print("Fitting LSA Encoder")
        print(f"{'='*60}")

        self.lsa_encoder = fit_lsa_encoder(
            emails,
            n_components=self.lsa_components,
            min_df=self.lsa_min_df,
            max_df=self.lsa_max_df
        )

        print(f"LSA encoder fitted with {self.lsa_encoder.n_components} components")

    def extract_features(self, emails: List[str]) -> np.ndarray:
        """
        Extract combined features (preprocessing + LSA) from emails.

        Args:
            emails: List of raw email texts

        Returns:
            Feature matrix of shape (n_emails, n_features)
        """
        if self.lsa_encoder is None:
            raise RuntimeError("LSA encoder not fitted. Call fit_lsa() first.")

        print(f"Extracting features from {len(emails)} emails...")

        # Use batch preprocessing with LSA
        results = preprocess_email_batch_with_lsa(emails, self.lsa_encoder)

        # Extract combined feature vectors
        X = np.array([r['combined_vector'] for r in results])

        if self.feature_dim is None:
            self.feature_dim = X.shape[1]

        print(f"Extracted features: shape = {X.shape}")
        print(f"  - {self.feature_dim} total dimensions")
        print(f"  - 34 numeric features + {self.lsa_encoder.n_components} LSA dimensions")

        return X

    def save_models(self, output_dir: Path):
        """
        Save the trained models and encoders.

        Args:
            output_dir: Directory to save models
        """
        output_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print("Saving Models")
        print(f"{'='*60}")

        # Save LSA encoder
        lsa_path = output_dir / 'lsa_encoder.pkl'
        joblib.dump(self.lsa_encoder, lsa_path)
        print(f"✓ LSA encoder saved: {lsa_path}")

        # Save classifier
        clf_path = output_dir / 'phishing_classifier.pkl'
        joblib.dump(self.classifier, clf_path)
        print(f"✓ Classifier saved: {clf_path}")

        # Save pipeline metadata
        metadata = {
            'lsa_components': self.lsa_encoder.n_components,
            'feature_dim': self.feature_dim,
            'lsa_min_df': self.lsa_min_df,
            'lsa_max_df': self.lsa_max_df,
        }
        metadata_path = output_dir / 'pipeline_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        print(f"✓ Metadata saved: {metadata_path}")

        print(f"\nAll models saved to: {output_dir}")

    def load_models(self, model_dir: Path):
        """
        Load trained models from disk.

        Args:
            model_dir: Directory containing saved models
        """
        print(f"Loading models from {model_dir}...")

        # Add bert_base to path for lsa_tool import
        import sys
        bert_base_path = str(Path(__file__).parent.parent / 'bert_base')
        if bert_base_path not in sys.path:
            sys.path.insert(0, bert_base_path)

        # Load LSA encoder
        lsa_path = model_dir / 'lsa_encoder.pkl'
        self.lsa_encoder = joblib.load(lsa_path)
        print(f"✓ LSA encoder loaded")

        # Load classifier
        clf_path = model_dir / 'phishing_classifier.pkl'
        self.classifier = joblib.load(clf_path)
        print(f"✓ Classifier loaded")

        # Load metadata
        metadata_path = model_dir / 'pipeline_metadata.pkl'
        metadata = joblib.load(metadata_path)
        self.feature_dim = metadata['feature_dim']
        self.lsa_min_df = metadata['lsa_min_df']
        self.lsa_max_df = metadata['lsa_max_df']
        print(f"✓ Metadata loaded")

        print(f"Models loaded successfully!")
