"""
Feedback Store & API Tests
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from email_monitor.feedback import FeedbackStore
from api.main import app

HAM   = "From: alice@co.com\nSubject: Lunch\n\nSee you at noon."
PHISH = "From: bad@evil.com\nSubject: URGENT\n\nVerify now: http://bit.ly/x"


# ── FeedbackStore ─────────────────────────────────────────────────────────────

class TestFeedbackStore:
    def _store(self):
        return FeedbackStore(":memory:")

    def test_record_correction(self):
        store = self._store()
        store.record(PHISH, predicted="benign", correct_label="phishing", confidence=0.3)
        stats = store.stats()
        assert stats["total"] == 1
        assert stats["corrections"] == 1
        assert stats["confirmations"] == 0

    def test_record_confirmation(self):
        store = self._store()
        store.record(HAM, predicted="benign", correct_label="benign", confidence=0.9)
        stats = store.stats()
        assert stats["total"] == 1
        assert stats["corrections"] == 0
        assert stats["confirmations"] == 1

    def test_corrections_returns_only_wrong(self):
        store = self._store()
        store.record(HAM,   predicted="benign",   correct_label="benign",   confidence=0.9)
        store.record(PHISH, predicted="benign",   correct_label="phishing", confidence=0.3)
        corrections = store.corrections()
        assert len(corrections) == 1
        email_text, label = corrections[0]
        assert label == 1  # phishing
        assert "evil.com" in email_text

    def test_corrections_label_mapping(self):
        store = self._store()
        store.record(PHISH, predicted="phishing", correct_label="benign", confidence=0.9)
        corrections = store.corrections()
        assert corrections[0][1] == 0  # benign → 0

    def test_empty_store_stats(self):
        store = self._store()
        stats = store.stats()
        assert stats == {"total": 0, "corrections": 0, "confirmations": 0}

    def test_empty_corrections(self):
        store = self._store()
        assert store.corrections() == []

    def test_recent(self):
        store = self._store()
        store.record(HAM,   predicted="benign",   correct_label="benign",   confidence=0.9)
        store.record(PHISH, predicted="benign",   correct_label="phishing", confidence=0.3)
        rows = store.recent(limit=10)
        assert len(rows) == 2
        # newest first
        assert rows[0]["correct_label"] == "phishing"

    def test_multiple_corrections(self):
        store = self._store()
        for i in range(5):
            store.record(f"email {i}", predicted="benign",
                         correct_label="phishing", confidence=0.2)
        assert store.stats()["corrections"] == 5
        assert len(store.corrections()) == 5


# ── /api/feedback endpoint ────────────────────────────────────────────────────

def _make_client():
    """Build a TestClient with a mini trained pipeline (reuse from test_api)."""
    import numpy as np
    from pipeline import EmailPhishingPipeline, Trainer
    from unittest.mock import patch as _patch

    ham_emails   = [HAM]   * 40
    phish_emails = [PHISH] * 40
    emails = ham_emails + phish_emails
    labels = np.array([0] * 40 + [1] * 40)

    pipeline = EmailPhishingPipeline(lsa_components=10, lsa_min_df=1)
    pipeline.fit_lsa(emails)
    X = pipeline.extract_features(emails)
    Trainer().train_classifier(pipeline, X, labels)

    import api.main as api_module
    from api.main import app

    def _load_mini():
        api_module._pipeline = pipeline
        return True

    with _patch('api.main._load_pipeline', _load_mini):
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
    return client


class TestFeedbackAPI:
    @pytest.fixture(autouse=True)
    def _client(self):
        import api.main as api_module
        import numpy as np
        from pipeline import EmailPhishingPipeline, Trainer
        from unittest.mock import patch as _patch

        emails = [HAM] * 40 + [PHISH] * 40
        labels = np.array([0] * 40 + [1] * 40)
        pipeline = EmailPhishingPipeline(lsa_components=5, lsa_min_df=1)
        pipeline.fit_lsa(emails)
        X = pipeline.extract_features(emails)
        Trainer().train_classifier(pipeline, X, labels)

        def _load_mini():
            api_module._pipeline = pipeline
            return True

        with _patch('api.main._load_pipeline', _load_mini):
            with TestClient(app, raise_server_exceptions=False) as c:
                self.client = c
                yield

    def test_submit_correction_returns_success(self):
        resp = self.client.post('/api/feedback', json={
            "email_text":    PHISH,
            "predicted":     "benign",
            "correct_label": "phishing",
            "confidence":    0.3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["was_correct"] is False

    def test_submit_confirmation_was_correct_true(self):
        resp = self.client.post('/api/feedback', json={
            "email_text":    HAM,
            "predicted":     "benign",
            "correct_label": "benign",
            "confidence":    0.9,
        })
        assert resp.status_code == 200
        assert resp.json()["was_correct"] is True

    def test_missing_email_text_returns_400(self):
        resp = self.client.post('/api/feedback', json={
            "email_text": "",
            "predicted": "benign",
            "correct_label": "phishing",
        })
        assert resp.status_code == 400

    def test_invalid_predicted_returns_400(self):
        resp = self.client.post('/api/feedback', json={
            "email_text": HAM,
            "predicted": "unknown",
            "correct_label": "benign",
        })
        assert resp.status_code == 400

    def test_invalid_correct_label_returns_400(self):
        resp = self.client.post('/api/feedback', json={
            "email_text": HAM,
            "predicted": "benign",
            "correct_label": "spam",
        })
        assert resp.status_code == 400

    def test_feedback_stats_no_db(self):
        with patch('api.main._feedback_db') as mock_db:
            mock_db.exists.return_value = False
            resp = self.client.get('/api/feedback/stats')
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["db_exists"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
