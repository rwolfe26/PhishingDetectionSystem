"""
Training Module

Handles model training, evaluation, cross-validation, hyperparameter tuning,
classifier benchmarking, and error analysis.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)


class Trainer:
    """Handles training, evaluation, tuning, and benchmarking of classifiers."""

    # Feature names aligned with EmailFeatures.to_dict() order (44 total)
    NUMERIC_FEATURE_NAMES = [
        # URL features (0-9)
        'num_urls', 'num_unique_domains', 'has_ip_url', 'no_https_ratio',
        'avg_url_length', 'max_url_length', 'avg_path_depth', 'total_dots_in_urls',
        'has_at_symbol_url', 'has_suspicious_port',
        # Text features (10-13)
        'num_words', 'num_unique_words', 'num_chars', 'num_special_chars',
        # Keyword features (14-16)
        'num_urgent_keywords', 'num_credential_keywords', 'num_action_keywords',
        # Header features (17-22)
        'has_reply_to_mismatch', 'has_return_path_mismatch', 'num_received_hops',
        'has_suspicious_mailer', 'subject_has_urgent', 'subject_has_re_fw',
        # HTML features (23-28)
        'has_html', 'has_form', 'has_iframe', 'has_hidden_text',
        'num_external_links', 'link_text_url_mismatch',
        # Attachment features (29-31)
        'num_attachments', 'has_executable_attachment', 'has_archive_attachment',
        # Phishing-specific signals (32-41)
        'spf_dkim_fail', 'sender_domain_mismatch',
        'num_homograph_chars', 'brand_impersonation_score',
        'urgency_density', 'html_text_ratio',
        'num_shortener_urls', 'greeting_generic',
        'num_auth_keywords', 'subject_all_caps_ratio',
        # URL redirect resolution (42-43)
        'has_redirect_url', 'num_redirect_hops',
    ]

    @staticmethod
    def train_classifier(pipeline, X_train: np.ndarray, y_train: np.ndarray,
                         params: dict = None):
        """
        Train the phishing classifier.

        Args:
            pipeline: EmailPhishingPipeline instance
            X_train: Training features
            y_train: Training labels
            params: Optional RF hyperparameters (overrides defaults)
        """
        print(f"\n{'='*60}")
        print("Training Classifier")
        print(f"{'='*60}")

        default_params = dict(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        if params:
            default_params.update(params)

        pipeline.classifier = RandomForestClassifier(**default_params)

        print(f"Training Random Forest on {len(X_train)} samples...")
        pipeline.classifier.fit(X_train, y_train)
        print("Classifier trained successfully!")

        Trainer._print_feature_importances(pipeline)

    @staticmethod
    def _print_feature_importances(pipeline, top_n: int = 20):
        """Print the top-N most important features."""
        if not hasattr(pipeline.classifier, 'feature_importances_'):
            return

        importances = pipeline.classifier.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        n_numeric = len(Trainer.NUMERIC_FEATURE_NAMES)

        print(f"\nTop {top_n} Most Important Features:")
        for rank, idx in enumerate(top_indices, 1):
            if idx < n_numeric:
                name = Trainer.NUMERIC_FEATURE_NAMES[idx]
                print(f"  {rank:2d}. {name} (numeric): {importances[idx]:.4f}")
            else:
                print(f"  {rank:2d}. LSA dim {idx - n_numeric}: {importances[idx]:.4f}")

    @staticmethod
    def evaluate(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the classifier and return metrics.

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

        y_pred = pipeline.classifier.predict(X_test)
        y_proba = pipeline.classifier.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        auc = roc_auc_score(y_test, y_proba)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                Ham    Spam")
        print(f"Actual  Ham    {tn:5d}  {fp:5d}")
        print(f"        Spam   {fn:5d}  {tp:5d}")

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\nError Rates:")
        print(f"  False Positive Rate: {fpr:.4f}  ({fp} ham emails flagged as spam)")
        print(f"  False Negative Rate: {fnr:.4f}  ({fn} spam emails missed)")

        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam/Phishing']))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba,
        }

    @staticmethod
    def cross_validate(pipeline, X: np.ndarray, y: np.ndarray, cv: int = 5):
        """
        Run stratified k-fold cross-validation and print per-fold metrics.

        Args:
            pipeline: EmailPhishingPipeline instance (for its classifier params)
            X: Feature matrix
            y: Labels
            cv: Number of folds
        """
        print(f"\n{'='*60}")
        print(f"Stratified {cv}-Fold Cross-Validation")
        print(f"{'='*60}")

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1]

            acc = accuracy_score(y_val, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
            auc = roc_auc_score(y_val, y_proba)

            fold_metrics.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc})
            print(f"  Fold {fold}: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
                  f"F1={f1:.4f}  AUC={auc:.4f}")

        print(f"\n  Mean ± Std:")
        for metric in ('acc', 'prec', 'rec', 'f1', 'auc'):
            vals = [m[metric] for m in fold_metrics]
            print(f"  {metric.upper():5s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    @staticmethod
    def tune_hyperparameters(pipeline, X_train: np.ndarray, y_train: np.ndarray,
                             n_iter: int = 30) -> dict:
        """
        Randomised hyperparameter search over the Random Forest.

        Args:
            pipeline: EmailPhishingPipeline instance
            X_train: Training features
            y_train: Training labels
            n_iter: Number of random parameter combinations to try

        Returns:
            Best hyperparameters found
        """
        print(f"\n{'='*60}")
        print("Hyperparameter Tuning (RandomizedSearchCV)")
        print(f"{'='*60}")

        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'class_weight': ['balanced', None],
        }

        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            clf, param_dist,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            refit=True,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        print(f"\nBest F1 (CV): {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")

        pipeline.classifier = search.best_estimator_
        return search.best_params_

    @staticmethod
    def benchmark_classifiers(pipeline, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray):
        """
        Train and evaluate multiple classifiers side-by-side.

        Args:
            pipeline: EmailPhishingPipeline instance
            X_train / y_train: Training data
            X_test / y_test: Test data
        """
        print(f"\n{'='*60}")
        print("Classifier Benchmark")
        print(f"{'='*60}")

        try:
            from xgboost import XGBClassifier
            scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            xgb = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=scale_pos, eval_metric='logloss',
                random_state=42, n_jobs=-1, verbosity=0,
            )
        except ImportError:
            xgb = None

        classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, class_weight='balanced', max_iter=1000, random_state=42
            ),
            'Linear SVM (calibrated)': CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
        }
        if xgb is not None:
            classifiers['XGBoost'] = xgb

        results = {}
        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_proba)

            results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}

        print(f"\n{'='*60}")
        print("Benchmark Results")
        print(f"{'='*60}")
        header = f"{'Classifier':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
        print(header)
        print("-" * len(header))
        for name, m in sorted(results.items(), key=lambda x: -x[1]['f1']):
            print(f"{name:<28} {m['acc']:>6.4f} {m['prec']:>6.4f} {m['rec']:>6.4f} "
                  f"{m['f1']:>6.4f} {m['auc']:>6.4f}")

        # Set best classifier on pipeline
        best_name = max(results, key=lambda n: results[n]['f1'])
        print(f"\nBest classifier by F1: {best_name}")
        pipeline.classifier = classifiers[best_name]

        return results

    @staticmethod
    def run_error_analysis(pipeline, email_texts: List[str], X_test: np.ndarray,
                           y_test: np.ndarray, metrics: dict,
                           output_path: Path = Path('error_analysis.txt')):
        """
        Save misclassified emails to a file for manual review.

        Args:
            pipeline: EmailPhishingPipeline instance
            email_texts: Raw email text for each test sample
            X_test: Test feature matrix
            y_test: True labels
            metrics: Evaluation dict from evaluate()
            output_path: File to write error analysis to
        """
        y_pred = metrics['predictions']
        y_proba = metrics['probabilities']

        false_positives = [
            (i, email_texts[i], y_proba[i])
            for i in range(len(y_test))
            if y_test[i] == 0 and y_pred[i] == 1
        ]
        false_negatives = [
            (i, email_texts[i], y_proba[i])
            for i in range(len(y_test))
            if y_test[i] == 1 and y_pred[i] == 0
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"FALSE POSITIVES — Ham misclassified as Spam ({len(false_positives)})\n")
            f.write("-" * 70 + "\n")
            for idx, (_, text, prob) in enumerate(false_positives[:50], 1):
                f.write(f"\n--- FP #{idx} (confidence={prob:.3f}) ---\n")
                f.write(text[:800])
                f.write("\n")

            f.write("\n\n" + "=" * 70 + "\n")
            f.write(f"FALSE NEGATIVES — Spam missed by classifier ({len(false_negatives)})\n")
            f.write("-" * 70 + "\n")
            for idx, (_, text, prob) in enumerate(false_negatives[:50], 1):
                f.write(f"\n--- FN #{idx} (confidence={prob:.3f}) ---\n")
                f.write(text[:800])
                f.write("\n")

        print(f"\nError analysis saved to: {output_path}")
        print(f"  False positives: {len(false_positives)}")
        print(f"  False negatives: {len(false_negatives)}")
