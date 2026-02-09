"""
Email Phishing Detection Pipeline - CLI Entry Point

Command-line interface for training and predicting with the phishing detection pipeline.

Usage:
    python run_pipeline.py --train [options]
    python run_pipeline.py --predict <email_file>
"""

import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

from pipeline import EmailPhishingPipeline, DataLoader, Trainer, Predictor


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

    # Define dataset directories
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

    # Load dataset
    data_loader = DataLoader()
    emails, labels = data_loader.load_dataset(ham_dirs, spam_dirs)

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
    trainer = Trainer()
    trainer.train_classifier(pipeline, X_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(pipeline, X_test, y_test)

    # Save models
    output_dir = base_dir / 'models'
    pipeline.save_models(output_dir)

    print(f"\n{'='*60}")
    print("Pipeline Training Complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {output_dir}")
    print(f"To use for prediction: python run_pipeline.py --predict <email_file>")


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
    predictor = Predictor()
    prediction, probability = predictor.predict_single(pipeline, email)

    # Display result
    predictor.format_prediction_result(str(email_path), prediction, probability)


def main():
    parser = argparse.ArgumentParser(
        description='Email Phishing Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the pipeline
  python run_pipeline.py --train

  # Train with custom parameters
  python run_pipeline.py --train --lsa-components 256 --test-size 0.2

  # Predict a single email
  python run_pipeline.py --predict Datasets/spam/0000.7b1b73cf36cf9dbc3d64e3f2ee2b91f1
        """
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
