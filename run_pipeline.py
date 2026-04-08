"""
Email Phishing Detection Pipeline - CLI Entry Point

Command-line interface for training and predicting with the phishing detection pipeline.

Usage:
    python run_pipeline.py --train [options]
    python run_pipeline.py --predict <email_file>
    python run_pipeline.py --benchmark
    python run_pipeline.py --tune
"""

import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

from pipeline import EmailPhishingPipeline, DataLoader, Trainer, Predictor


def train_pipeline(args):
    """Train the complete pipeline."""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / 'Datasets'

    pipeline = EmailPhishingPipeline(
        lsa_components=args.lsa_components,
        lsa_min_df=args.lsa_min_df,
        lsa_max_df=args.lsa_max_df
    )

    # Collect all available ham and spam directories
    ham_dirs = [d for d in [
        dataset_dir / 'easy_ham',
        dataset_dir / 'easy_ham_2',
        dataset_dir / 'easy_ham_3',
        dataset_dir / 'hard_ham',
        dataset_dir / 'hard_ham_2',
    ] if d.exists()]

    spam_dirs = [d for d in [
        dataset_dir / 'spam',
        dataset_dir / 'spam_2',
        dataset_dir / 'spam_3',
        dataset_dir / 'spam_4',
    ] if d.exists()]

    print(f"\n{'='*60}")
    print("Loading Dataset")
    print(f"{'='*60}")
    print(f"Ham directories ({len(ham_dirs)}): {[d.name for d in ham_dirs]}")
    print(f"Spam directories ({len(spam_dirs)}): {[d.name for d in spam_dirs]}")

    import numpy as np

    data_loader = DataLoader()
    emails, dir_labels = data_loader.load_dataset(ham_dirs, spam_dirs)
    all_labels = list(dir_labels)

    # Also load phishing emails from CSV if available
    phishing_csv = dataset_dir / 'Phishing_Email.csv'
    if phishing_csv.exists():
        csv_emails, csv_labels = data_loader.load_phishing_csv(phishing_csv, max_samples=args.csv_samples)
        emails.extend(csv_emails)
        all_labels.extend(csv_labels)
        print(f"Added {len(csv_emails)} emails from Phishing_Email.csv")

    # Also load JSONL phishing data if available
    jsonl_file = dataset_dir / 'phishing and benign email dataset.jsonl'
    if jsonl_file.exists():
        jsonl_emails, jsonl_labels = data_loader.load_jsonl_phishing(jsonl_file)
        emails.extend(jsonl_emails)
        all_labels.extend(jsonl_labels)
        print(f"Added {len(jsonl_emails)} emails from JSONL dataset")

    labels = np.array(all_labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(emails)
    print(f"\nTotal dataset: {total} emails")
    for u, c in zip(unique, counts):
        label_name = "Ham" if u == 0 else "Phishing/Spam"
        print(f"  {label_name}: {c} ({c/total*100:.1f}%)")

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

    # Single-pass: fit LSA + extract training features (2× faster)
    X_train = pipeline.fit_lsa_and_extract(X_train_emails)

    # Extract test features (uses already-fitted LSA)
    print(f"\n{'='*60}")
    print("Test Set Feature Extraction")
    print(f"{'='*60}")
    X_test = pipeline.extract_features(X_test_emails)

    # Train classifier
    trainer = Trainer()

    if args.cross_validate:
        trainer.cross_validate(pipeline, X_train, y_train, cv=args.cv_folds)

    trainer.train_classifier(pipeline, X_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(pipeline, X_test, y_test)

    if args.error_analysis:
        trainer.run_error_analysis(pipeline, X_test_emails, X_test, y_test, metrics)

    # Save models
    output_dir = base_dir / 'models'
    pipeline.save_models(output_dir)

    print(f"\n{'='*60}")
    print("Pipeline Training Complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {output_dir}")
    print(f"To predict: python run_pipeline.py --predict <email_file>")


def predict_email(args):
    """Predict a single email using trained models."""
    base_dir = Path(__file__).parent
    model_dir = base_dir / 'models'

    pipeline = EmailPhishingPipeline()
    pipeline.load_models(model_dir)

    email_path = Path(args.predict)
    if not email_path.exists():
        print(f"Error: Email file not found: {email_path}")
        return

    with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
        email = f.read()

    predictor = Predictor()
    prediction, probability = predictor.predict_single(pipeline, email)
    predictor.format_prediction_result(str(email_path), prediction, probability)


def benchmark_classifiers(args):
    """Benchmark multiple classifiers on the dataset."""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / 'Datasets'

    pipeline = EmailPhishingPipeline(lsa_components=args.lsa_components)

    ham_dirs = [d for d in [
        dataset_dir / 'easy_ham',
        dataset_dir / 'easy_ham_2',
        dataset_dir / 'hard_ham',
    ] if d.exists()]
    spam_dirs = [d for d in [
        dataset_dir / 'spam',
        dataset_dir / 'spam_2',
    ] if d.exists()]

    data_loader = DataLoader()
    emails, labels = data_loader.load_dataset(ham_dirs, spam_dirs)

    phishing_csv = dataset_dir / 'Phishing_Email.csv'
    if phishing_csv.exists():
        csv_emails, csv_labels = data_loader.load_phishing_csv(phishing_csv, max_samples=2000)
        emails.extend(csv_emails)
        labels.extend(csv_labels)

    import numpy as np
    labels = np.array(labels)

    X_train_emails, X_test_emails, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline.fit_lsa(X_train_emails)
    X_train = pipeline.extract_features(X_train_emails)
    X_test = pipeline.extract_features(X_test_emails)

    trainer = Trainer()
    trainer.benchmark_classifiers(pipeline, X_train, y_train, X_test, y_test)


def tune_pipeline(args):
    """Run hyperparameter tuning for the Random Forest."""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / 'Datasets'

    pipeline = EmailPhishingPipeline(lsa_components=args.lsa_components)

    ham_dirs = [d for d in [dataset_dir / 'easy_ham', dataset_dir / 'easy_ham_2'] if d.exists()]
    spam_dirs = [d for d in [dataset_dir / 'spam', dataset_dir / 'spam_2'] if d.exists()]

    data_loader = DataLoader()
    emails, labels = data_loader.load_dataset(ham_dirs, spam_dirs)

    import numpy as np
    labels = np.array(labels)

    X_train_emails, X_test_emails, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline.fit_lsa(X_train_emails)
    X_train = pipeline.extract_features(X_train_emails)
    X_test = pipeline.extract_features(X_test_emails)

    trainer = Trainer()
    best_params = trainer.tune_hyperparameters(pipeline, X_train, y_train)

    print(f"\nRetraining with best params: {best_params}")
    pipeline.classifier = None
    trainer.train_classifier(pipeline, X_train, y_train)
    trainer.evaluate(pipeline, X_test, y_test)


def main():
    parser = argparse.ArgumentParser(
        description='Email Phishing Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --train
  python run_pipeline.py --train --cross-validate --error-analysis
  python run_pipeline.py --train --lsa-components 256 --test-size 0.2
  python run_pipeline.py --benchmark
  python run_pipeline.py --tune
  python run_pipeline.py --predict Datasets/spam/0000.7b1b73cf36cf9dbc3d64e3f2ee2b91f1
        """
    )

    # Mode selection
    parser.add_argument('--train', action='store_true', help='Train the pipeline')
    parser.add_argument('--predict', type=str, help='Predict a single email (path to file)')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark multiple classifiers')
    parser.add_argument('--tune', action='store_true', help='Tune Random Forest hyperparameters')

    # Training parameters
    parser.add_argument('--lsa-components', type=int, default=256,
                        help='Number of LSA components (default: 256)')
    parser.add_argument('--lsa-min-df', type=int, default=2,
                        help='Minimum document frequency for LSA (default: 2)')
    parser.add_argument('--lsa-max-df', type=float, default=0.85,
                        help='Maximum document frequency for LSA (default: 0.85)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--csv-samples', type=int, default=5000,
                        help='Max samples to load from Phishing_Email.csv (default: 5000)')

    # Evaluation options
    parser.add_argument('--cross-validate', action='store_true',
                        help='Run stratified k-fold cross-validation before final training')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--error-analysis', action='store_true',
                        help='Run error analysis and save misclassified emails to error_analysis.txt')

    args = parser.parse_args()

    if args.train:
        train_pipeline(args)
    elif args.predict:
        predict_email(args)
    elif args.benchmark:
        benchmark_classifiers(args)
    elif args.tune:
        tune_pipeline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
