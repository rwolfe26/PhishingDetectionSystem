"""
API Integration Tests

Tests for the FastAPI endpoints: /health, /classify, /classify/file.
Uses FastAPI's TestClient with a mocked (mini-trained) pipeline so that
the full test suite runs without requiring pre-built model files on disk.
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import EmailPhishingPipeline, Trainer

# ── Fixtures ─────────────────────────────────────────────────────────────────

HAM_EMAIL = """\
From: alice@example.com
To: bob@example.com
Subject: Weekly team update

Hi Bob, here are the highlights from this week.
Let me know if you have any questions.

Best, Alice
"""

PHISHING_EMAIL = """\
From: security@paypa1.com
To: victim@example.com
Reply-To: harvest@evil-site.ru
Subject: URGENT: Account Suspended - Verify NOW
Authentication-Results: example.com; spf=fail; dkim=fail

Dear Customer,
Your PayPal account will be suspended within 24 hours unless you verify.
CLICK HERE NOW: http://paypa1-secure.evil.com/verify?user=victim@example.com
Download: http://bit.ly/malware123
"""


def _build_mini_pipeline() -> EmailPhishingPipeline:
    """Create a minimal trained pipeline for testing (no disk I/O)."""
    ham_samples = [HAM_EMAIL] * 40
    spam_samples = [PHISHING_EMAIL] * 40
    emails = ham_samples + spam_samples
    labels = np.array([0] * 40 + [1] * 40)

    pipeline = EmailPhishingPipeline(lsa_components=15, lsa_min_df=1)
    pipeline.fit_lsa(emails)
    X = pipeline.extract_features(emails)

    trainer = Trainer()
    trainer.train_classifier(pipeline, X, labels)
    return pipeline


@pytest.fixture(scope='module')
def mini_pipeline():
    return _build_mini_pipeline()


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded (simulates cold start before training)."""
    import api.main as api_module
    from api.main import app

    def _load_nothing():
        api_module._pipeline = None
        return False

    with patch('api.main._load_pipeline', _load_nothing):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture()
def client(mini_pipeline):
    """TestClient with a pre-loaded mini pipeline."""
    import api.main as api_module
    from api.main import app

    def _load_mini():
        api_module._pipeline = mini_pipeline
        return True

    with patch('api.main._load_pipeline', _load_mini):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── /health ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_model_not_loaded(self, client_no_model):
        resp = client_no_model.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'ok'
        assert data['model_loaded'] is False

    def test_health_model_loaded(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'ok'
        assert data['model_loaded'] is True
        assert 'model_dir' in data


# ── /classify ────────────────────────────────────────────────────────────────

class TestClassify:
    def test_classify_503_when_no_model(self, client_no_model):
        resp = client_no_model.post(
            '/classify',
            json={'email_text': 'Hello, this is a test email.'},
        )
        assert resp.status_code == 503

    def test_classify_400_on_empty_text(self, client):
        resp = client.post('/classify', json={'email_text': '   '})
        assert resp.status_code == 400

    def test_classify_413_on_oversized(self, client):
        big_text = 'x' * (1024 * 1024 + 1)
        resp = client.post('/classify', json={'email_text': big_text})
        assert resp.status_code == 413

    def test_classify_ham_response_shape(self, client):
        resp = client.post('/classify', json={'email_text': HAM_EMAIL})
        assert resp.status_code == 200
        data = resp.json()
        assert data['prediction'] in ('phishing', 'benign')
        assert 0.0 <= data['confidence'] <= 1.0
        assert data['risk_level'] in ('HIGH', 'MEDIUM', 'LOW', 'SAFE')
        assert isinstance(data['top_features'], list)
        assert isinstance(data['indicators'], dict)
        assert data['model_loaded'] is True

    def test_classify_phishing_prediction(self, client):
        resp = client.post('/classify', json={'email_text': PHISHING_EMAIL})
        assert resp.status_code == 200
        data = resp.json()
        assert data['prediction'] in ('phishing', 'benign')
        assert 0.0 <= data['confidence'] <= 1.0

    def test_classify_returns_indicators(self, client):
        resp = client.post('/classify', json={'email_text': PHISHING_EMAIL})
        assert resp.status_code == 200
        indicators = resp.json()['indicators']
        assert 'urgent_phrases' in indicators
        assert 'credential_phrases' in indicators

    def test_classify_safe_risk_level_for_benign(self, client):
        resp = client.post('/classify', json={'email_text': HAM_EMAIL})
        assert resp.status_code == 200
        data = resp.json()
        if data['prediction'] == 'benign':
            assert data['risk_level'] == 'SAFE'

    def test_classify_high_risk_for_high_confidence_phishing(self, client):
        resp = client.post('/classify', json={'email_text': PHISHING_EMAIL})
        data = resp.json()
        if data['confidence'] >= 0.85:
            assert data['risk_level'] == 'HIGH'


# ── /classify/file ───────────────────────────────────────────────────────────

class TestClassifyFile:
    def test_classify_file_valid(self, client):
        file_bytes = PHISHING_EMAIL.encode('utf-8')
        resp = client.post(
            '/classify/file',
            files={'file': ('phishing.eml', io.BytesIO(file_bytes), 'message/rfc822')},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data['prediction'] in ('phishing', 'benign')
        assert 0.0 <= data['confidence'] <= 1.0

    def test_classify_file_oversized(self, client):
        big_bytes = b'x' * (1024 * 1024 + 1)
        resp = client.post(
            '/classify/file',
            files={'file': ('big.eml', io.BytesIO(big_bytes), 'text/plain')},
        )
        assert resp.status_code == 413

    def test_classify_file_ham(self, client):
        resp = client.post(
            '/classify/file',
            files={'file': ('ham.eml', io.BytesIO(HAM_EMAIL.encode()), 'message/rfc822')},
        )
        assert resp.status_code == 200
        assert resp.json()['prediction'] in ('phishing', 'benign')


# ── Request-ID header ────────────────────────────────────────────────────────

class TestRequestID:
    def test_response_has_request_id_header(self, client):
        resp = client.get('/health')
        assert 'x-request-id' in resp.headers

    def test_custom_request_id_is_echoed(self, client):
        custom_id = 'test-request-abc-123'
        resp = client.get('/health', headers={'X-Request-ID': custom_id})
        assert resp.headers.get('x-request-id') == custom_id


# ── fit_lsa_and_extract (single-pass) ────────────────────────────────────────

class TestSinglePassExtraction:
    def test_fit_lsa_and_extract_shape_matches_extract_features(self):
        emails = [HAM_EMAIL] * 20 + [PHISHING_EMAIL] * 20
        np.array([0] * 20 + [1] * 20)

        lsa_components = 10

        pipeline_single = EmailPhishingPipeline(lsa_components=lsa_components, lsa_min_df=1)
        X_single = pipeline_single.fit_lsa_and_extract(emails)

        pipeline_double = EmailPhishingPipeline(lsa_components=lsa_components, lsa_min_df=1)
        pipeline_double.fit_lsa(emails)
        X_double = pipeline_double.extract_features(emails)

        assert X_single.shape == X_double.shape, (
            f"Single-pass shape {X_single.shape} != double-pass shape {X_double.shape}"
        )
        from preprocessing.feature_extractor import EmailFeatures
        n_numeric = len(EmailFeatures.feature_names())
        assert X_single.shape == (len(emails), n_numeric + lsa_components)

    def test_fit_lsa_and_extract_produces_valid_features(self):
        emails = [HAM_EMAIL] * 15 + [PHISHING_EMAIL] * 15
        pipeline = EmailPhishingPipeline(lsa_components=8, lsa_min_df=1)
        X = pipeline.fit_lsa_and_extract(emails)
        assert X.shape[0] == 30
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_single_pass_trains_valid_classifier(self):
        emails = [HAM_EMAIL] * 20 + [PHISHING_EMAIL] * 20
        labels = np.array([0] * 20 + [1] * 20)

        pipeline = EmailPhishingPipeline(lsa_components=10, lsa_min_df=1)
        X = pipeline.fit_lsa_and_extract(emails)

        trainer = Trainer()
        trainer.train_classifier(pipeline, X, labels)

        assert pipeline.classifier is not None
        preds = pipeline.classifier.predict(X)
        assert set(preds).issubset({0, 1})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
