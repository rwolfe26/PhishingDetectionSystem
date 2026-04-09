"""
Complete Email Phishing Detection Pipeline

This script implements the full end-to-end pipeline:
1. Load raw emails (ham and spam)
2. Fit LSA encoder on training data
3. Extract combined features (preprocessing + LSA)
4. Train classifier
5. Evaluate model
6. Save all artifacts for deployment

Usage:
    python pipeline.py --train
    python pipeline.py --predict <email_file>
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# Import our integrated preprocessing + LSA functions
from preprocessing import (
    fit_lsa_encoder,
    preprocess_email_batch_with_lsa,
    preprocess_email_with_lsa,
)


class EmailPhishingPipeline:
    """Complete pipeline for email phishing detection."""

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

    def load_emails_from_directory(self, directory: Path, label: int) -> Tuple[List[str], List[int]]:
        """
        Load emails from a directory and assign labels.

        Args:
            directory: Path to directory containing email files
            label: Label to assign (0=ham, 1=spam)

        Returns:
            Tuple of (emails, labels)
        """
        emails = []
        labels = []

        if not directory.exists():
            print(f"Warning: Directory not found: {directory}")
            return emails, labels

        for filename in os.listdir(directory):
            filepath = directory / filename
            if filepath.is_file():
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        email_content = f.read()
                        emails.append(email_content)
                        labels.append(label)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        return emails, labels

    def load_dataset(self,
                    ham_dirs: List[Path],
                    spam_dirs: List[Path]) -> Tuple[List[str], np.ndarray]:
        """
        Load complete dataset from multiple directories.

        Args:
            ham_dirs: List of directories containing ham emails
            spam_dirs: List of directories containing spam emails

        Returns:
            Tuple of (emails, labels)
        """
        all_emails = []
        all_labels = []

        # Load ham emails (label=0)
        for ham_dir in ham_dirs:
            print(f"Loading ham emails from {ham_dir}...")
            emails, labels = self.load_emails_from_directory(ham_dir, label=0)
            all_emails.extend(emails)
            all_labels.extend(labels)
            print(f"  Loaded {len(emails)} ham emails")

        # Load spam emails (label=1)
        for spam_dir in spam_dirs:
            print(f"Loading spam emails from {spam_dir}...")
            emails, labels = self.load_emails_from_directory(spam_dir, label=1)
            all_emails.extend(emails)
            all_labels.extend(labels)
            print(f"  Loaded {len(emails)} spam emails")

        print(f"\nTotal dataset: {len(all_emails)} emails")
        print(f"  Ham: {sum(1 for l in all_labels if l == 0)}")
        print(f"  Spam: {sum(1 for l in all_labels if l == 1)}")

        return all_emails, np.array(all_labels)

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

    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the phishing classifier.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print("Training Classifier")
        print(f"{'='*60}")

        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        print(f"Training on {len(X_train)} samples...")
        self.classifier.fit(X_train, y_train)

        print("Classifier trained successfully!")

        # Show feature importance (top 20)
        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
            top_indices = np.argsort(importances)[-20:][::-1]

            print("\nTop 20 Most Important Features:")
            for i, idx in enumerate(top_indices, 1):
                if idx < 34:
                    print(f"  {i}. Feature {idx} (numeric): {importances[idx]:.4f}")
                else:
                    print(f"  {i}. LSA dimension {idx-34}: {importances[idx]:.4f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the classifier.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call train_classifier() first.")

        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")

        # Make predictions
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                Ham    Spam")
        print(f"Actual  Ham    {tn:5d}  {fp:5d}")
        print(f"        Spam   {fn:5d}  {tp:5d}")

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Ham', 'Spam']))

        # False positives and false negatives analysis
        print("\nError Analysis:")
        print(f"  False Positives (Ham marked as Spam): {fp}")
        print(f"  False Negatives (Spam marked as Ham): {fn}")
        print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
        print(f"  False Negative Rate: {fn/(fn+tp):.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }

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
        bert_base_path = str(Path(__file__).parent / 'bert_base')
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

    def predict_single(self, email: str) -> Tuple[int, float]:
        """
        Predict whether a single email is spam.

        Args:
            email: Raw email text

        Returns:
            Tuple of (prediction, probability)
            prediction: 0=ham, 1=spam
            probability: confidence score for spam class
        """
        if self.lsa_encoder is None or self.classifier is None:
            raise RuntimeError("Models not loaded. Call load_models() or train first.")

        # Extract features
        result = preprocess_email_with_lsa(email, self.lsa_encoder)
        X = result['combined_vector'].reshape(1, -1)

        # Predict
        prediction = self.classifier.predict(X)[0]
        probability = self.classifier.predict_proba(X)[0][1]  # Probability of spam

        return prediction, probability


def train_pipeline(args):
    """Train the complete pipeline."""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / 'Datasets'

    # Initialize pipeline
    pipeline = EmailPhishingPipeline(
        lsa_components=args.lsa_components,
        lsa_min_df=args.lsa_min_df,
        lsa_max_df=args.lsa_max_df
    )

    # Load dataset
    ham_dirs = [
        dataset_dir / 'easy_ham',
        # Add more ham directories here if available
        # dataset_dir / 'easy_ham_2',
        # dataset_dir / 'hard_ham',
    ]
    spam_dirs = [
        dataset_dir / 'spam',
        # Add more spam directories here if available
        # dataset_dir / 'spam_2',
    ]

    emails, labels = pipeline.load_dataset(ham_dirs, spam_dirs)

    # Split into train/test
    print(f"\n{'='*60}")
    print("Splitting Dataset")
    print(f"{'='*60}")
    X_train_emails, X_test_emails, y_train, y_test = train_test_split(
        emails, labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels
    )
    print(f"Training set: {len(X_train_emails)} emails")
    print(f"Test set: {len(X_test_emails)} emails")

    # Fit LSA on training data only
    pipeline.fit_lsa(X_train_emails)

    # Extract features
    print(f"\n{'='*60}")
    print("Feature Extraction")
    print(f"{'='*60}")
    X_train = pipeline.extract_features(X_train_emails)
    X_test = pipeline.extract_features(X_test_emails)

    # Train classifier
    pipeline.train_classifier(X_train, y_train)

    # Evaluate
    pipeline.evaluate(X_test, y_test)

    # Save models
    output_dir = base_dir / 'models'
    pipeline.save_models(output_dir)

    print(f"\n{'='*60}")
    print("Pipeline Training Complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {output_dir}")
    print("To use for prediction: python pipeline.py --predict <email_file>")


def predict_email(args):
    """Predict a single email using trained models."""
    base_dir = Path(__file__).parent
    model_dir = base_dir / 'models'

    # Load pipeline
    pipeline = EmailPhishingPipeline()
    pipeline.load_models(model_dir)

    # Load email
    email_path = Path(args.predict)
    if not email_path.exists():
        print(f"Error: Email file not found: {email_path}")
        return

    with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
        email = f.read()

    # Predict
    prediction, probability = pipeline.predict_single(email)

    # Display result
    print(f"\n{'='*60}")
    print("Prediction Result")
    print(f"{'='*60}")
    print(f"Email: {email_path}")
    print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
    print(f"Confidence: {probability:.2%}")

    if prediction == 1:
        if probability > 0.9:
            print("Risk Level: HIGH (Very likely spam)")
        elif probability > 0.7:
            print("Risk Level: MEDIUM (Likely spam)")
        else:
            print("Risk Level: LOW (Possibly spam)")
    else:
        print("Risk Level: Safe (Legitimate email)")


def main():
    parser = argparse.ArgumentParser(
        description='Email Phishing Detection Pipeline'
    )

    # Mode selection
    parser.add_argument('--train', action='store_true',
                       help='Train the pipeline')
    parser.add_argument('--predict', type=str,
                       help='Predict a single email (path to email file)')

    # Training parameters
    parser.add_argument('--lsa-components', type=int, default=768,
                       help='Number of LSA components (default: 768)')
    parser.add_argument('--lsa-min-df', type=int, default=2,
                       help='Minimum document frequency for LSA (default: 2)')
    parser.add_argument('--lsa-max-df', type=float, default=0.85,
                       help='Maximum document frequency for LSA (default: 0.85)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')

    args = parser.parse_args()

    if args.train:
        train_pipeline(args)
    elif args.predict:
        predict_email(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
