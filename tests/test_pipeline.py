"""
Pipeline Integration Tests

Tests for the full EmailPhishingPipeline, DataLoader, Trainer, and Predictor.
Covers training, prediction, model persistence, edge cases, and new features.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import EmailPhishingPipeline, DataLoader, Trainer, Predictor
from pipeline.trainer import Trainer as TrainerClass
from preprocessing import preprocess_email, fit_lsa_encoder, preprocess_email_with_lsa
from preprocessing.feature_extractor import FeatureExtractor, EmailFeatures


# ── Fixtures / helpers ───────────────────────────────────────────────────────

HAM_EMAIL = """\
From: alice@example.com
To: bob@example.com
Subject: Weekly team update
Date: Mon, 07 Apr 2026 10:00:00 +0000
Message-ID: <test-001@example.com>

Hi Bob,

Here are the key highlights from this week:
- Completed the quarterly review
- Scheduled product demos for Thursday
- Code review for PR #42 is done

Let me know if you have any questions.

Best,
Alice
"""

PHISHING_EMAIL = """\
From: security@paypa1.com
To: victim@example.com
Reply-To: harvest@evil-site.ru
Subject: URGENT: Account Suspended - Verify NOW
Date: Mon, 07 Apr 2026 09:00:00 +0000
Authentication-Results: example.com; spf=fail; dkim=fail; dmarc=fail

Dear Customer,

Your PayPal account will be permanently suspended within 24 hours unless you
verify your identity immediately.

CLICK HERE NOW: http://paypa1-secure.evil.com/verify?user=victim@example.com

Please enter your password, credit card and social security number to
confirm your account. Act now - this is your final notice!

Download the verification tool: http://bit.ly/malware123

- PayPal Security Team
"""

EMPTY_EMAIL = ""
HTML_ONLY_EMAIL = """\
From: newsletter@legit.com
Subject: Your weekly digest
Content-Type: text/html; charset=utf-8
MIME-Version: 1.0

<html><body><h1>Hello</h1><p>Check out this week's updates.</p></body></html>
"""

ATTACHMENT_EMAIL = """\
From: attacker@evil.com
Subject: Invoice attached
Content-Type: multipart/mixed; boundary="boundary"
MIME-Version: 1.0

--boundary
Content-Type: text/plain

Please find the invoice attached.

--boundary
Content-Disposition: attachment; filename="invoice.exe"
Content-Type: application/octet-stream

FAKEEXECUTABLECONTENT

--boundary--
"""

DISPLAY_NAME_PHISHING_EMAIL = """\
From: PayPal Security <attacker@totally-evil.ru>
To: victim@example.com
Subject: Account suspended - act now

Dear Customer, your PayPal account has been suspended.
Verify now: http://paypa1-verify.com/account
"""


def _make_mini_pipeline(lsa_components: int = 20) -> tuple:
    """Create a trained mini-pipeline for testing."""
    # Generate synthetic training data (fast)
    ham_samples = [HAM_EMAIL] * 30 + [HTML_ONLY_EMAIL] * 20
    spam_samples = [PHISHING_EMAIL] * 30 + [ATTACHMENT_EMAIL] * 20

    emails = ham_samples + spam_samples
    labels = np.array([0] * 50 + [1] * 50)

    pipeline = EmailPhishingPipeline(lsa_components=lsa_components, lsa_min_df=1)
    pipeline.fit_lsa(emails)
    X = pipeline.extract_features(emails)

    trainer = Trainer()
    trainer.train_classifier(pipeline, X, labels)

    return pipeline, emails, labels, X


# ── Feature extractor tests ──────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_ham_email_features(self):
        result = preprocess_email(HAM_EMAIL)
        # 'features' is an EmailFeatures object; 'feature_vector' is the list
        assert isinstance(result['feature_vector'], list)
        assert len(result['feature_vector']) == len(EmailFeatures.feature_names())

    def test_phishing_email_features(self):
        result = preprocess_email(PHISHING_EMAIL)
        d = result['feature_dict']  # note: feature_dict, not features_dict

        assert d['has_reply_to_mismatch'] == 1, "Reply-To mismatch not detected"
        assert d['num_urgent_keywords'] > 0, "Urgency keywords not found"
        assert d['num_credential_keywords'] > 0, "Credential keywords not found"
        assert d['subject_has_urgent'] == 1, "Subject urgency not detected"

    def test_phishing_new_features(self):
        from preprocessing.email_parser import parse_email as ep
        from preprocessing.url_extractor import URLExtractor
        from preprocessing.feature_extractor import FeatureExtractor

        # PHISHING_EMAIL has SPF fail, shortener, generic greeting, auth keywords
        parsed = ep(PHISHING_EMAIL)
        urls = URLExtractor().extract_all(text=parsed.body_text, html=parsed.body_html)
        features = FeatureExtractor().extract(parsed, urls)

        assert features.spf_dkim_fail == 1, "SPF/DKIM fail not detected"
        assert features.num_shortener_urls >= 1, "URL shortener not detected"
        assert features.greeting_generic == 1, "Generic greeting not detected"
        assert features.num_auth_keywords > 0, "Auth keywords not found"

        # Use display-name phishing email for sender domain mismatch test
        parsed2 = ep(DISPLAY_NAME_PHISHING_EMAIL)
        urls2 = URLExtractor().extract_all(text=parsed2.body_text, html=parsed2.body_html)
        features2 = FeatureExtractor().extract(parsed2, urls2)
        assert features2.sender_domain_mismatch == 1, (
            "Sender domain mismatch not detected for display-name spoofing"
        )

    def test_feature_names_count(self):
        names = EmailFeatures.feature_names()
        assert len(names) == 47, f"Expected 47 features, got {len(names)}"

    def test_empty_email(self):
        result = preprocess_email(EMPTY_EMAIL)
        assert isinstance(result['feature_vector'], list)
        assert len(result['feature_vector']) == 47

    def test_html_only_email(self):
        result = preprocess_email(HTML_ONLY_EMAIL)
        assert result['feature_dict']['has_html'] == 1

    def test_attachment_detection(self):
        from preprocessing.email_parser import parse_email as ep
        from preprocessing.feature_extractor import FeatureExtractor

        parsed = ep(ATTACHMENT_EMAIL)
        features = FeatureExtractor().extract(parsed)
        assert features.has_executable_attachment == 1

    def test_brand_impersonation_score(self):
        from preprocessing.feature_extractor import FeatureExtractor
        ext = FeatureExtractor()
        # "paypa1" should be very close to "paypal" → high impersonation score
        dist = ext._levenshtein("paypa1", "paypal")
        assert dist <= 2, f"Expected distance ≤ 2, got {dist}"

    def test_levenshtein_identical(self):
        ext = FeatureExtractor()
        assert ext._levenshtein("paypal", "paypal") == 0

    def test_levenshtein_different(self):
        ext = FeatureExtractor()
        assert ext._levenshtein("", "abc") == 3
        assert ext._levenshtein("abc", "") == 3


# ── LSA tests ────────────────────────────────────────────────────────────────

class TestLSAEncoder:
    def test_fit_and_transform(self):
        emails = [HAM_EMAIL, PHISHING_EMAIL, HTML_ONLY_EMAIL] * 5
        lsa = fit_lsa_encoder(emails, n_components=10, min_df=1)
        vecs = lsa.transform(emails)
        assert vecs.shape == (len(emails), 10)

    def test_combined_vector_shape(self):
        emails = [HAM_EMAIL, PHISHING_EMAIL] * 10
        lsa = fit_lsa_encoder(emails, n_components=15, min_df=1)
        result = preprocess_email_with_lsa(HAM_EMAIL, lsa)
        vec = result['combined_vector']
        n_numeric = len(EmailFeatures.feature_names())
        assert vec.shape == (n_numeric + 15,), f"Expected ({n_numeric + 15},), got {vec.shape}"

    def test_transform_single(self):
        # Use diverse emails and min_df=1 with enough unique tokens
        emails = [HAM_EMAIL, PHISHING_EMAIL, HTML_ONLY_EMAIL,
                  DISPLAY_NAME_PHISHING_EMAIL] * 5
        lsa = fit_lsa_encoder(emails, n_components=5, min_df=1, max_df=1.0)
        result = preprocess_email_with_lsa(PHISHING_EMAIL, lsa)
        assert 'combined_vector' in result
        assert 'lsa_embedding' in result


# ── DataLoader tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_from_directory(self, tmp_path):
        # Write fake emails
        for i in range(5):
            (tmp_path / f"ham_{i}.eml").write_text(HAM_EMAIL, encoding='utf-8')

        loader = DataLoader()
        emails, labels = loader.load_emails_from_directory(tmp_path, label=0)
        assert len(emails) == 5
        assert all(l == 0 for l in labels)

    def test_load_dataset(self, tmp_path):
        ham_dir = tmp_path / 'ham'
        spam_dir = tmp_path / 'spam'
        ham_dir.mkdir()
        spam_dir.mkdir()

        for i in range(5):
            (ham_dir / f"h{i}.eml").write_text(HAM_EMAIL)
        for i in range(3):
            (spam_dir / f"s{i}.eml").write_text(PHISHING_EMAIL)

        loader = DataLoader()
        emails, labels = loader.load_dataset([ham_dir], [spam_dir])
        assert len(emails) == 8
        assert (labels == 0).sum() == 5
        assert (labels == 1).sum() == 3

    def test_missing_directory(self):
        loader = DataLoader()
        emails, labels = loader.load_emails_from_directory(
            Path('/nonexistent/path'), label=0
        )
        assert emails == []
        assert labels == []

    def test_load_jsonl(self, tmp_path):
        import json
        jsonl = tmp_path / 'phishing.jsonl'
        with open(jsonl, 'w') as f:
            f.write(json.dumps({'subject': 'test', 'body': 'verify now', 'label': 'phishing'}) + '\n')
            f.write(json.dumps({'subject': 'hello', 'body': 'just an email', 'label': 'benign'}) + '\n')

        loader = DataLoader()
        emails, labels = loader.load_jsonl_phishing(jsonl)
        assert len(emails) == 2
        assert 1 in labels
        assert 0 in labels

    def test_load_phishing_csv(self, tmp_path):
        csv_file = tmp_path / 'phishing.csv'
        csv_file.write_text(
            ',Email Text,Email Type\n'
            '0,"phishing email text here","Phishing Email"\n'
            '1,"safe email text here","Safe Email"\n',
            encoding='utf-8'
        )
        loader = DataLoader()
        emails, labels = loader.load_phishing_csv(csv_file)
        assert len(emails) == 2
        assert 1 in labels
        assert 0 in labels


# ── Pipeline integration tests ───────────────────────────────────────────────

class TestPipeline:
    def test_full_train_predict_cycle(self):
        pipeline, emails, labels, X = _make_mini_pipeline()
        assert pipeline.classifier is not None
        assert X.shape[0] == len(emails)

        predictor = Predictor()
        pred, prob = predictor.predict_single(pipeline, PHISHING_EMAIL)
        assert pred in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_save_and_load_models(self, tmp_path):
        pipeline, _, _, _ = _make_mini_pipeline()
        pipeline.save_models(tmp_path)

        assert (tmp_path / 'lsa_encoder.pkl').exists()
        assert (tmp_path / 'phishing_classifier.pkl').exists()
        assert (tmp_path / 'pipeline_metadata.pkl').exists()

        pipeline2 = EmailPhishingPipeline()
        pipeline2.load_models(tmp_path)
        assert pipeline2.classifier is not None

        pred, prob = Predictor().predict_single(pipeline2, PHISHING_EMAIL)
        assert pred in (0, 1)

    def test_feature_dim_consistency(self):
        pipeline, emails, _, X = _make_mini_pipeline(lsa_components=15)
        n_numeric = len(EmailFeatures.feature_names())
        assert X.shape[1] == n_numeric + 15  # numeric features + 15 LSA dims

    def test_empty_email_prediction(self):
        pipeline, _, _, _ = _make_mini_pipeline()
        predictor = Predictor()
        pred, prob = predictor.predict_single(pipeline, EMPTY_EMAIL)
        assert pred in (0, 1)
        assert 0.0 <= prob <= 1.0

    def test_html_email_prediction(self):
        pipeline, _, _, _ = _make_mini_pipeline()
        predictor = Predictor()
        pred, prob = predictor.predict_single(pipeline, HTML_ONLY_EMAIL)
        assert pred in (0, 1)


# ── Trainer tests ────────────────────────────────────────────────────────────

class TestTrainer:
    def test_evaluate_returns_metrics(self):
        pipeline, _, labels, X = _make_mini_pipeline()
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=42)
        trainer = Trainer()
        trainer.train_classifier(pipeline, X_tr, y_tr)
        metrics = trainer.evaluate(pipeline, X_te, y_te)
        for key in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_cross_validate(self, capsys):
        pipeline, _, labels, X = _make_mini_pipeline()
        trainer = Trainer()
        trainer.cross_validate(pipeline, X, labels, cv=3)
        captured = capsys.readouterr()
        assert 'Mean' in captured.out or 'Fold' in captured.out

    def test_error_analysis(self, tmp_path):
        pipeline, emails, labels, X = _make_mini_pipeline()
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=42)
        emails_test = [emails[i] for i in range(len(y_te))]

        trainer = Trainer()
        metrics = trainer.evaluate(pipeline, X_te, y_te)
        output_path = tmp_path / 'errors.txt'
        trainer.run_error_analysis(pipeline, emails_test, X_te, y_te, metrics, output_path)
        assert output_path.exists()

    def test_feature_names_match_n_numeric(self):
        assert len(TrainerClass.NUMERIC_FEATURE_NAMES) == 47


# ── Explainability tests ─────────────────────────────────────────────────────

class TestExplainability:
    def test_explain_prediction(self):
        from api.explainer import explain_prediction
        pipeline, _, _, X = _make_mini_pipeline()
        explanations = explain_prediction(pipeline, X[0], top_n=5)
        assert isinstance(explanations, list)
        assert len(explanations) <= 5
        if explanations:
            exp = explanations[0]
            assert 'name' in exp
            assert 'description' in exp
            assert 'direction' in exp

    def test_highlight_indicators(self):
        from api.explainer import highlight_phishing_indicators
        import numpy as np
        vec = np.zeros(42 + 20)
        indicators = highlight_phishing_indicators(PHISHING_EMAIL, vec)
        assert isinstance(indicators, dict)
        assert 'urgent_phrases' in indicators
        assert len(indicators['urgent_phrases']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
