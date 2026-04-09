"""
Email Monitor Tests

Tests for email_monitor components using mocks — no real IMAP connection
or trained model required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from email_monitor.storage import ClassificationStore
from email_monitor.notifier import alert_phishing, alert_benign, print_status
from email_monitor.monitor import EmailMonitor, _risk_level

# ── Sample emails ─────────────────────────────────────────────────────────────

HAM = """\
From: alice@company.com
To: bob@company.com
Message-ID: <ham-001@company.com>
Subject: Team lunch Friday
Date: Wed, 8 Apr 2026 10:00:00 +0000

Hi Bob, lunch is at noon on Friday.
"""

PHISH = """\
From: security@paypa1-alert.com
To: victim@gmail.com
Message-ID: <phish-001@evil.com>
Reply-To: harvest@evil.ru
Subject: URGENT: Account Suspended
Date: Wed, 8 Apr 2026 10:01:00 +0000

Dear Customer, verify now: http://bit.ly/verify123
Your account will be suspended within 24 hours.
"""


# ── Helper: build a monitor with a fully mocked pipeline ─────────────────────

def _mock_pipeline(prediction: int = 1, probability: float = 0.92):
    """Return a mock pipeline that always returns the given prediction."""
    pipeline = MagicMock()
    pipeline.classifier = MagicMock()
    pipeline.lsa_encoder = MagicMock()
    pipeline.classifier.predict.return_value = [prediction]
    pipeline.classifier.predict_proba.return_value = [[1 - probability, probability]]
    return pipeline


def _make_monitor(pipeline=None, db_path=":memory:"):
    if pipeline is None:
        pipeline = _mock_pipeline()
    return EmailMonitor(
        pipeline=pipeline,
        host="imap.example.com",
        user="test@example.com",
        password="secret",
        db_path=db_path,
    )


# ── ClassificationStore ───────────────────────────────────────────────────────

class TestClassificationStore:
    def test_save_and_retrieve(self):
        store = ClassificationStore(":memory:")
        store.save({
            "message_id": "<test-1@example.com>",
            "subject": "Test",
            "sender": "sender@example.com",
            "prediction": "phishing",
            "confidence": 0.95,
            "risk_level": "HIGH",
        })
        recent = store.recent(limit=5)
        assert len(recent) == 1
        assert recent[0]["prediction"] == "phishing"
        assert recent[0]["confidence"] == 0.95

    def test_already_seen(self):
        store = ClassificationStore(":memory:")
        msg_id = "<duplicate@example.com>"
        store.save({
            "message_id": msg_id,
            "prediction": "benign",
            "confidence": 0.9,
            "risk_level": "SAFE",
        })
        assert store.already_seen(msg_id) is True
        assert store.already_seen("<other@example.com>") is False

    def test_duplicate_insert_ignored(self):
        store = ClassificationStore(":memory:")
        record = {
            "message_id": "<dup@example.com>",
            "prediction": "phishing",
            "confidence": 0.8,
            "risk_level": "HIGH",
        }
        store.save(record)
        store.save(record)  # second insert should be silently ignored
        assert len(store.recent()) == 1

    def test_stats(self):
        store = ClassificationStore(":memory:")
        store.save({"message_id": "<a@x.com>", "prediction": "phishing", "confidence": 0.9, "risk_level": "HIGH"})
        store.save({"message_id": "<b@x.com>", "prediction": "benign",   "confidence": 0.85, "risk_level": "SAFE"})
        store.save({"message_id": "<c@x.com>", "prediction": "phishing", "confidence": 0.7, "risk_level": "MEDIUM"})
        stats = store.stats()
        assert stats["total"] == 3
        assert stats["phishing"] == 2
        assert stats["benign"] == 1

    def test_top_features_stored_as_json(self):
        store = ClassificationStore(":memory:")
        features = [{"feature": "urgency_density", "value": 0.05}]
        store.save({
            "message_id": "<feat@x.com>",
            "prediction": "phishing",
            "confidence": 0.88,
            "risk_level": "HIGH",
            "top_features": features,
        })
        row = store.recent(1)[0]
        assert json.loads(row["top_features"]) == features

    def test_already_seen_empty_message_id(self):
        store = ClassificationStore(":memory:")
        assert store.already_seen("") is False


# ── Risk level helper ─────────────────────────────────────────────────────────

class TestRiskLevel:
    def test_high(self):
        assert _risk_level(0.90) == "HIGH"
        assert _risk_level(0.85) == "HIGH"

    def test_medium(self):
        assert _risk_level(0.75) == "MEDIUM"
        assert _risk_level(0.60) == "MEDIUM"

    def test_low(self):
        assert _risk_level(0.50) == "LOW"
        assert _risk_level(0.40) == "LOW"

    def test_safe(self):
        assert _risk_level(0.10) == "SAFE"
        assert _risk_level(0.00) == "SAFE"


# ── EmailMonitor._parse_headers ───────────────────────────────────────────────

class TestParseHeaders:
    def setup_method(self):
        self.monitor = _make_monitor()

    def test_parses_standard_headers(self):
        h = self.monitor._parse_headers(PHISH)
        assert h["subject"] == "URGENT: Account Suspended"
        assert "paypa1-alert.com" in h["sender"]
        assert h["message_id"] == "<phish-001@evil.com>"

    def test_handles_missing_headers(self):
        h = self.monitor._parse_headers("No headers here\n\nJust a body.")
        assert h["subject"] == "(no subject)"
        assert h["sender"] == "(unknown)"

    def test_handles_empty_string(self):
        h = self.monitor._parse_headers("")
        assert isinstance(h, dict)


# ── EmailMonitor._classify ────────────────────────────────────────────────────

class TestClassify:
    def test_phishing_prediction(self):
        monitor = _make_monitor(_mock_pipeline(prediction=1, probability=0.92))
        with patch("email_monitor.monitor.Predictor.predict_single", return_value=(1, 0.92)):
            result = monitor._classify(PHISH)
        assert result["prediction"] == "phishing"
        assert result["confidence"] == pytest.approx(0.92, abs=0.01)
        assert result["risk_level"] == "HIGH"

    def test_benign_prediction(self):
        monitor = _make_monitor(_mock_pipeline(prediction=0, probability=0.08))
        with patch("email_monitor.monitor.Predictor.predict_single", return_value=(0, 0.08)):
            result = monitor._classify(HAM)
        assert result["prediction"] == "benign"
        assert result["risk_level"] == "SAFE"

    def test_result_has_required_keys(self):
        with patch("email_monitor.monitor.Predictor.predict_single", return_value=(1, 0.75)):
            result = _make_monitor()._classify(PHISH)
        for key in ("prediction", "confidence", "risk_level", "top_features"):
            assert key in result


# ── EmailMonitor.scan_once ────────────────────────────────────────────────────

class TestScanOnce:
    def _run_scan(self, emails, prediction=1, probability=0.92):
        monitor = _make_monitor()
        with patch("email_monitor.monitor.IMAPClient") as MockIMAP, \
             patch("email_monitor.monitor.Predictor.predict_single",
                   return_value=(prediction, probability)):
            instance = MockIMAP.return_value.__enter__.return_value
            instance.fetch_unseen.return_value = emails
            count = monitor.scan_once()
        return count, monitor.store

    def test_phishing_email_stored(self):
        count, store = self._run_scan([("1", PHISH)], prediction=1, probability=0.92)
        assert count == 1
        recent = store.recent()
        assert recent[0]["prediction"] == "phishing"

    def test_benign_email_stored(self):
        count, store = self._run_scan([("1", HAM)], prediction=0, probability=0.05)
        assert count == 1
        recent = store.recent()
        assert recent[0]["prediction"] == "benign"

    def test_empty_mailbox_returns_zero(self):
        count, _ = self._run_scan([])
        assert count == 0

    def test_duplicate_not_stored_twice(self):
        monitor = _make_monitor()
        emails = [("1", PHISH)]
        with patch("email_monitor.monitor.IMAPClient") as MockIMAP, \
             patch("email_monitor.monitor.Predictor.predict_single",
                   return_value=(1, 0.92)):
            instance = MockIMAP.return_value.__enter__.return_value
            instance.fetch_unseen.return_value = emails
            monitor.scan_once()
            monitor.scan_once()  # second scan should skip the same message
        assert monitor.store.stats()["total"] == 1

    def test_classification_error_does_not_crash(self):
        monitor = _make_monitor()
        with patch("email_monitor.monitor.IMAPClient") as MockIMAP, \
             patch("email_monitor.monitor.Predictor.predict_single",
                   side_effect=RuntimeError("model error")):
            instance = MockIMAP.return_value.__enter__.return_value
            instance.fetch_unseen.return_value = [("1", PHISH)]
            count = monitor.scan_once()  # should not raise
        assert count == 0  # failed classification, nothing stored

    def test_multiple_emails_processed(self):
        count, store = self._run_scan(
            [("1", PHISH), ("2", HAM)],
            prediction=1, probability=0.9,
        )
        assert count == 2
        assert store.stats()["total"] == 2


# ── Notifier (smoke tests — just ensure no exceptions) ────────────────────────

class TestNotifier:
    def test_alert_phishing_no_crash(self, capsys):
        alert_phishing(
            subject="URGENT verify",
            sender="bad@evil.com",
            confidence=0.93,
            risk_level="HIGH",
            top_features=[{"feature": "urgency_density"}],
        )
        out = capsys.readouterr().out
        assert "PHISHING" in out

    def test_alert_benign_no_crash(self, capsys):
        alert_benign(subject="Lunch plans", sender="alice@co.com", confidence=0.9)
        out = capsys.readouterr().out
        assert "SAFE" in out

    def test_print_status_no_crash(self, capsys):
        print_status("Testing 1 2 3")
        out = capsys.readouterr().out
        assert "Testing 1 2 3" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
