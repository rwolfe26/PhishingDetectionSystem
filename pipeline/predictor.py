"""
Prediction Module

Handles prediction logic for classifying new emails.
"""

from typing import Tuple
from preprocessing import preprocess_email_with_lsa


class Predictor:
    """Handles prediction on new emails."""

    @staticmethod
    def predict_single(pipeline, email: str) -> Tuple[int, float]:
        """
        Predict whether a single email is spam.

        Args:
            pipeline: EmailPhishingPipeline instance
            email: Raw email text

        Returns:
            Tuple of (prediction, probability)
            prediction: 0=ham, 1=spam
            probability: confidence score for spam class
        """
        if pipeline.lsa_encoder is None or pipeline.classifier is None:
            raise RuntimeError("Models not loaded. Call load_models() or train first.")

        # Extract features
        result = preprocess_email_with_lsa(email, pipeline.lsa_encoder)
        X = result['combined_vector'].reshape(1, -1)

        # Predict
        prediction = pipeline.classifier.predict(X)[0]
        probability = pipeline.classifier.predict_proba(X)[0][1]  # Probability of spam

        return prediction, probability

    @staticmethod
    def format_prediction_result(email_path: str, prediction: int, probability: float):
        """
        Format prediction result for display.

        Args:
            email_path: Path to the email file
            prediction: Prediction (0=ham, 1=spam)
            probability: Confidence score
        """
        print(f"\n{'='*60}")
        print("Prediction Result")
        print(f"{'='*60}")
        print(f"Email: {email_path}")
        print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
        print(f"Confidence: {probability:.2%}")

        if prediction == 1:
            if probability > 0.9:
                print(f"Risk Level: HIGH (Very likely spam)")
            elif probability > 0.7:
                print(f"Risk Level: MEDIUM (Likely spam)")
            else:
                print(f"Risk Level: LOW (Possibly spam)")
        else:
            print(f"Risk Level: Safe (Legitimate email)")
