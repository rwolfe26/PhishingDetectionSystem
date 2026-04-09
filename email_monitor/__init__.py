"""
Email Monitor — Phase 2

Polls an IMAP mailbox, classifies every unseen email for phishing,
stores results in SQLite, and prints colour-coded terminal alerts.
"""

from .imap_client import IMAPClient
from .storage import ClassificationStore
from .monitor import EmailMonitor

__all__ = ["EmailMonitor", "ClassificationStore", "IMAPClient"]
