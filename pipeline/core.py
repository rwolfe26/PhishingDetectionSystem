"""
Core Pipeline Class

Main EmailPhishingPipeline class that orchestrates the complete workflow.
"""

from pathlib import Path
from typing import List
import numpy as np
import joblib

from preprocessing import (
    fit_lsa_encoder,
    fit_lsa_encoder_from_texts,
)


class EmailPhishingPipeline:
    """
    Complete pipeline for email phishing detection.

    Combines preprocessing (42 numeric features) with LSA semantic analysis
    to create rich feature vectors for classification.
    """

    def __init__(self,
                 lsa_components: int = 128,
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

        Note: Prefer fit_lsa_and_extract() to avoid double-preprocessing.

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

    def fit_lsa_and_extract(self, emails: List[str]) -> np.ndarray:
        """
        Fit the LSA encoder AND extract features in a single preprocessing pass.

        This is 2× faster than calling fit_lsa() followed by extract_features(),
        because it avoids running the full email parser twice.

        Args:
            emails: List of raw email texts (training set)

        Returns:
            Feature matrix of shape (n_emails, n_numeric + n_lsa)
        """

        print(f"\n{'='*60}")
        print("Single-Pass: Preprocessing + LSA Fit + Feature Extraction")
        print(f"{'='*60}")
        print(f"Processing {len(emails)} emails (single pass)...")

        # One pass: parse emails, collect body texts and numeric features
        body_texts = []
        numeric_features_list = []
        for i, email in enumerate(emails):
            from preprocessing import preprocess_email as _pe
            result = _pe(email)
            body_texts.append(f"{result['subject']} {result['body_text']}")
            numeric_features_list.append(np.array(result['feature_vector'], dtype=np.float32))
            if (i + 1) % 2000 == 0:
                print(f"  Parsed {i+1}/{len(emails)} emails...")

        # Fit LSA on body texts
        self.lsa_encoder = fit_lsa_encoder_from_texts(
            body_texts,
            n_components=self.lsa_components,
            min_df=self.lsa_min_df,
            max_df=self.lsa_max_df,
        )

        # Batch transform all texts with LSA at once
        lsa_embeddings = self.lsa_encoder.transform(body_texts)

        # Concatenate numeric + LSA for each email
        X = np.array([
            np.concatenate([numeric_features_list[i], lsa_embeddings[i]])
            for i in range(len(emails))
        ])

        self.feature_dim = X.shape[1]
        n_numeric = len(numeric_features_list[0])
        print(f"Feature extraction complete: shape = {X.shape}")
        print(f"  - {n_numeric} numeric features + {self.lsa_encoder.n_components} LSA dims")

        return X

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

        from preprocessing import preprocess_email as _pe
        body_texts = []
        numeric_features_list = []
        for i, email in enumerate(emails):
            result = _pe(email)
            body_texts.append(f"{result['subject']} {result['body_text']}")
            numeric_features_list.append(np.array(result['feature_vector'], dtype=np.float32))
            if (i + 1) % 2000 == 0:
                print(f"  Parsed {i+1}/{len(emails)} emails...")

        lsa_embeddings = self.lsa_encoder.transform(body_texts)

        X = np.array([
            np.concatenate([numeric_features_list[i], lsa_embeddings[i]])
            for i in range(len(emails))
        ])

        if self.feature_dim is None:
            self.feature_dim = X.shape[1]

        n_numeric = len(numeric_features_list[0]) if numeric_features_list else 0
        print(f"Extracted features: shape = {X.shape}")
        print(f"  - {n_numeric} numeric features + {self.lsa_encoder.n_components} LSA dims")

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
        print("✓ LSA encoder loaded")

        # Load classifier
        clf_path = model_dir / 'phishing_classifier.pkl'
        self.classifier = joblib.load(clf_path)
        print("✓ Classifier loaded")

        # Load metadata
        metadata_path = model_dir / 'pipeline_metadata.pkl'
        metadata = joblib.load(metadata_path)
        self.feature_dim = metadata['feature_dim']
        self.lsa_min_df = metadata['lsa_min_df']
        self.lsa_max_df = metadata['lsa_max_df']
        print("✓ Metadata loaded")

        print("Models loaded successfully!")
