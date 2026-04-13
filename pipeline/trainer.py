"""
Training Module

Handles model training and evaluation logic.
"""

from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)


class Trainer:
    """Handles training and evaluation of the classifier."""

    @staticmethod
    def train_classifier(pipeline, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the phishing classifier.

        Args:
            pipeline: EmailPhishingPipeline instance
            X_train: Training features
            y_train: Training labels
        """
        print(f"\n{'='*60}")
        print("Training Classifier")
        print(f"{'='*60}")

        pipeline.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        print(f"Training on {len(X_train)} samples...")
        pipeline.classifier.fit(X_train, y_train)

        print("Classifier trained successfully!")

        # Show feature importance (top 20)
        if hasattr(pipeline.classifier, 'feature_importances_'):
            importances = pipeline.classifier.feature_importances_
            top_indices = np.argsort(importances)[-20:][::-1]

            print("\nTop 20 Most Important Features:")
            for i, idx in enumerate(top_indices, 1):
                if idx < 34:
                    print(f"  {i}. Feature {idx} (numeric): {importances[idx]:.4f}")
                else:
                    print(f"  {i}. LSA dimension {idx-34}: {importances[idx]:.4f}")

    @staticmethod
    def evaluate(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the classifier.

        Args:
            pipeline: EmailPhishingPipeline instance
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if pipeline.classifier is None:
            raise RuntimeError("Classifier not trained. Call train_classifier() first.")

        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")

        # Make predictions
        y_pred = pipeline.classifier.predict(X_test)
        y_proba = pipeline.classifier.predict_proba(X_test)

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

        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                Ham    Spam")
        print(f"Actual  Ham    {tn:5d}  {fp:5d}")
        print(f"        Spam   {fn:5d}  {tp:5d}")

        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Ham', 'Spam']))

        # False positives and false negatives analysis
        print(f"\nError Analysis:")
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
