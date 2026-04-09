"""
IMAP Client

Handles connecting to an IMAP server and fetching unseen emails.
Uses only the Python standard library (imaplib).
"""

import imaplib
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class IMAPClient:
    """
    Thin wrapper around imaplib that connects to an IMAP server,
    fetches UNSEEN messages, and returns their raw RFC-822 text.

    The mailbox is opened read-only so this tool never marks emails
    as read, moves them, or modifies the mailbox in any way.
    """

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        port: int = 993,
        use_ssl: bool = True,
        mailbox: str = "INBOX",
    ):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.use_ssl = use_ssl
        self.mailbox = mailbox
        self._conn: imaplib.IMAP4 | None = None

    # ── Connection lifecycle ─────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the connection and authenticate."""
        if self.use_ssl:
            self._conn = imaplib.IMAP4_SSL(self.host, self.port)
        else:
            self._conn = imaplib.IMAP4(self.host, self.port)
        self._conn.login(self.user, self.password)
        logger.info("Connected to %s as %s", self.host, self.user)

    def disconnect(self) -> None:
        """Close the connection gracefully."""
        if self._conn:
            try:
                self._conn.logout()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "IMAPClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Fetching ─────────────────────────────────────────────────────────────

    def fetch_unseen(self) -> List[Tuple[str, str]]:
        """
        Return all UNSEEN messages in the monitored mailbox.

        Returns:
            List of (uid, raw_email_text) tuples.
            uid is the IMAP UID as a string.
            raw_email_text is the full RFC-822 message as a string.
        """
        if self._conn is None:
            raise RuntimeError("Not connected — call connect() first.")

        # Open read-only so we never affect the seen/unseen state in the server
        self._conn.select(self.mailbox, readonly=True)

        typ, data = self._conn.search(None, "UNSEEN")
        if typ != "OK" or not data[0]:
            return []

        results: List[Tuple[str, str]] = []
        for uid_bytes in data[0].split():
            uid = uid_bytes.decode()
            typ, msg_data = self._conn.fetch(uid_bytes, "(RFC822)")
            if typ != "OK" or not msg_data or msg_data[0] is None:
                logger.warning("Failed to fetch uid %s", uid)
                continue
            raw = msg_data[0][1]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            results.append((uid, raw))

        logger.info("Fetched %d unseen message(s) from %s", len(results), self.mailbox)
        return results
