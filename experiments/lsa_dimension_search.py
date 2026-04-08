"""
LSA Dimensionality Search

Tests multiple LSA component counts and measures their impact on
downstream classification accuracy, F1, and AUC-ROC.

Usage:
    python experiments/lsa_dimension_search.py
    python experiments/lsa_dimension_search.py --max-emails 3000
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from pipeline import DataLoader
from preprocessing import fit_lsa_encoder, preprocess_email_batch_with_lsa


def run_experiment(emails, labels, n_components: int, seed: int = 42) -> dict:
    """
    Run a single LSA dimensionality experiment.

    Fits LSA with `n_components` on training data, extracts combined features,
    trains a Random Forest, and evaluates on a held-out test set.
    """
    X_train_emails, X_test_emails, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # Fit LSA on train only
    lsa = fit_lsa_encoder(X_train_emails, n_components=n_components, min_df=2, max_df=0.85)

    # Extract features
    train_results = preprocess_email_batch_with_lsa(X_train_emails, lsa)
    test_results = preprocess_email_batch_with_lsa(X_test_emails, lsa)

    X_train = np.array([r['combined_vector'] for r in train_results])
    X_test = np.array([r['combined_vector'] for r in test_results])

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight='balanced',
        random_state=seed, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_proba)

    return {
        'n_components': n_components,
        'feature_dim': X_train.shape[1],
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
    }


def main():
    parser = argparse.ArgumentParser(description='LSA Dimensionality Search')
    parser.add_argument('--max-emails', type=int, default=None,
                        help='Cap total emails per class for faster runs')
    parser.add_argument('--components', nargs='+', type=int,
                        default=[25, 50, 100, 150, 200, 256, 384, 512, 768],
                        help='LSA component counts to test')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / 'Datasets'

    ham_dirs = [d for d in [
        dataset_dir / 'easy_ham',
        dataset_dir / 'easy_ham_2',
        dataset_dir / 'hard_ham',
    ] if d.exists()]
    spam_dirs = [d for d in [
        dataset_dir / 'spam',
        dataset_dir / 'spam_2',
    ] if d.exists()]

    loader = DataLoader()
    emails, labels = loader.load_dataset(ham_dirs, spam_dirs)

    # Optional cap for speed
    if args.max_emails:
        from sklearn.utils import resample
        ham_idx = [i for i, l in enumerate(labels) if l == 0][:args.max_emails]
        spam_idx = [i for i, l in enumerate(labels) if l == 1][:args.max_emails]
        keep = ham_idx + spam_idx
        emails = [emails[i] for i in keep]
        labels = labels[keep]

    print(f"\nDataset: {len(emails)} emails")
    print(f"Testing LSA components: {args.components}\n")

    results = []
    for n in args.components:
        print(f"Testing n_components={n}...", end=' ', flush=True)
        result = run_experiment(emails, labels, n_components=n)
        results.append(result)
        print(f"F1={result['f1']:.4f}  AUC={result['auc']:.4f}  "
              f"dim={result['feature_dim']}")

    print(f"\n{'='*70}")
    print("LSA Dimensionality Search Results")
    print(f"{'='*70}")
    header = (f"{'n_components':>12} {'feat_dim':>9} {'Accuracy':>9} "
              f"{'Precision':>10} {'Recall':>7} {'F1':>7} {'AUC':>7}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['n_components']:>12} {r['feature_dim']:>9} {r['accuracy']:>9.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['auc']:>7.4f}")

    best = max(results, key=lambda r: r['f1'])
    print(f"\nBest by F1: n_components={best['n_components']}  "
          f"F1={best['f1']:.4f}  AUC={best['auc']:.4f}")

    # Save results to CSV
    output_path = base_dir / 'experiments' / 'lsa_dimension_results.csv'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('n_components,feature_dim,accuracy,precision,recall,f1,auc\n')
        for r in results:
            f.write(f"{r['n_components']},{r['feature_dim']},{r['accuracy']:.6f},"
                    f"{r['precision']:.6f},{r['recall']:.6f},{r['f1']:.6f},{r['auc']:.6f}\n")
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
