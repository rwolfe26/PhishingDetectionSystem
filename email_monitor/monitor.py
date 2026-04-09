"""
Email Monitor

Core monitoring loop: poll IMAP → classify → store → alert.
"""

import email as _email_stdlib
import logging
import time
from typing import Dict, Any

from pipeline.predictor import Predictor
from .imap_client import IMAPClient
from .storage import ClassificationStore
from . import notifier

logger = logging.getLogger(__name__)

# Risk thresholds (phishing probability)
_HIGH   = 0.85
_MEDIUM = 0.60
_LOW    = 0.40


def _risk_level(phishing_prob: float) -> str:
    if phishing_prob >= _HIGH:
        return "HIGH"
    if phishing_prob >= _MEDIUM:
        return "MEDIUM"
    if phishing_prob >= _LOW:
        return "LOW"
    return "SAFE"


class EmailMonitor:
    """
    Polls an IMAP mailbox at a fixed interval, runs every unseen email
    through the phishing classifier, and persists results to SQLite.

    Quick start::

        from pipeline import EmailPhishingPipeline
        from email_monitor import EmailMonitor

        pipeline = EmailPhishingPipeline.load("./models")
        monitor = EmailMonitor(pipeline, host="imap.gmail.com",
                               user="you@gmail.com", password="app_pw")
        monitor.run(interval=60)

    Gmail requires an *App Password* (not your regular password).
    See: myaccount.google.com/apppasswords
    """

    def __init__(
        self,
        pipeline,
        host: str,
        user: str,
        password: str,
        port: int = 993,
        mailbox: str = "INBOX",
        db_path: str = "monitor.db",
    ):
        self.pipeline = pipeline
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.mailbox = mailbox
        self.store = ClassificationStore(db_path)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _parse_headers(self, raw: str) -> Dict[str, str]:
        """Extract the key envelope headers from a raw RFC-822 email."""
        try:
            msg = _email_stdlib.message_from_string(raw)
            return {
                "message_id": (msg.get("Message-ID") or "").strip(),
                "subject":    (msg.get("Subject")    or "(no subject)").strip(),
                "sender":     (msg.get("From")       or "(unknown)").strip(),
                "received_at":(msg.get("Date")       or "").strip(),
            }
        except Exception as exc:
            logger.debug("Header parse error: %s", exc)
            return {"message_id": "", "subject": "", "sender": "", "received_at": ""}

    def _classify(self, raw: str) -> Dict[str, Any]:
        """Run the pipeline on a raw email and return a result dict."""
        prediction_int, prob = Predictor.predict_single(self.pipeline, raw)
        prediction = "phishing" if prediction_int == 1 else "benign"

        # prob is always the phishing-class probability
        phishing_prob = float(prob)
        confidence = phishing_prob if prediction == "phishing" else 1.0 - phishing_prob

        # Retrieve top contributing features if available
        top_features: list = []
        try:
            from api.explainer import Explainer
            top_features = Explainer.top_features(self.pipeline, raw, n=5)
        except Exception:
            pass

        return {
            "prediction":   prediction,
            "confidence":   round(confidence, 4),
            "risk_level":   _risk_level(phishing_prob),
            "top_features": top_features,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def scan_once(self) -> int:
        """
        Connect to IMAP, fetch unseen emails, classify and store each one.

        Returns:
            Number of emails newly processed in this scan.
        """
        processed = 0

        with IMAPClient(
            self.host, self.user, self.password,
            port=self.port, mailbox=self.mailbox,
        ) as client:
            emails = client.fetch_unseen()

        if not emails:
            logger.debug("No unseen emails.")
            return 0

        notifier.print_status(
            f"Found {len(emails)} unseen email(s) — classifying..."
        )

        for uid, raw in emails:
            headers = self._parse_headers(raw)
            msg_id = headers["message_id"] or f"imap-uid:{uid}"

            if self.store.already_seen(msg_id):
                logger.debug("Skipping already-seen message %s", msg_id)
                continue

            try:
                result = self._classify(raw)
            except Exception as exc:
                logger.error("Classification failed for uid %s: %s", uid, exc)
                notifier.print_warning(
                    f"Could not classify uid {uid} ({headers['subject'][:40]}): {exc}"
                )
                continue

            record = {**headers, **result, "imap_uid": uid, "message_id": msg_id}
            self.store.save(record)

            if result["prediction"] == "phishing":
                notifier.alert_phishing(
                    headers["subject"],
                    headers["sender"],
                    result["confidence"],
                    result["risk_level"],
                    result["top_features"],
                )
            else:
                notifier.alert_benign(
                    headers["subject"],
                    headers["sender"],
                    result["confidence"],
                )

            processed += 1

        return processed

    def run(self, interval: int = 60) -> None:
        """
        Poll the mailbox in a loop until the user presses Ctrl+C.

        Args:
            interval: Seconds to wait between scans (default 60).
        """
        notifier.print_status(
            f"Monitoring {self.user} @ {self.host}:{self.port}/{self.mailbox}"
        )
        notifier.print_status(
            f"Polling every {interval}s  •  Press Ctrl+C to stop.\n"
        )

        try:
            while True:
                try:
                    self.scan_once()
                except Exception as exc:
                    logger.error("Scan error: %s", exc)
                    notifier.print_warning(f"Scan failed: {exc}")
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            notifier.print_summary(self.store.stats())
