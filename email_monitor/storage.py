"""
Classification Store

SQLite-backed persistence layer for email classification history.
Uses only the Python standard library (sqlite3, json).
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List


class ClassificationStore:
    """
    Persists every classified email to a local SQLite database.

    The store is safe for single-process use.  Duplicate emails
    (identified by their Message-ID header) are silently ignored so
    re-scanning a mailbox never creates duplicate rows.
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS classifications (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id    TEXT    UNIQUE,
            subject       TEXT,
            sender        TEXT,
            received_at   TEXT,
            classified_at TEXT    NOT NULL,
            prediction    TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            risk_level    TEXT    NOT NULL,
            top_features  TEXT,
            imap_uid      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pred ON classifications (prediction);
        CREATE INDEX IF NOT EXISTS idx_time ON classifications (classified_at);
    """

    def __init__(self, db_path: str = "monitor.db"):
        self.db_path = db_path
        # Use a single persistent connection so that :memory: databases
        # (used in tests) aren't recreated on every query.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        return self._conn

    def _init_db(self) -> None:
        self._conn.executescript(self._DDL)

    # ── Write ────────────────────────────────────────────────────────────────

    def save(self, record: Dict[str, Any]) -> int:
        """
        Persist a classification result.

        Args:
            record: dict containing at minimum:
                prediction, confidence, risk_level.
                Optional: message_id, subject, sender, received_at,
                          top_features, imap_uid.

        Returns:
            rowid of the inserted row, or 0 if it was a duplicate.
        """
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO classifications
                    (message_id, subject, sender, received_at, classified_at,
                     prediction, confidence, risk_level, top_features, imap_uid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("message_id") or None,
                    record.get("subject", ""),
                    record.get("sender", ""),
                    record.get("received_at", ""),
                    datetime.now(timezone.utc).isoformat(),
                    record["prediction"],
                    float(record["confidence"]),
                    record["risk_level"],
                    json.dumps(record.get("top_features") or []),
                    record.get("imap_uid"),
                ),
            )
            return cur.lastrowid or 0

    # ── Read ─────────────────────────────────────────────────────────────────

    def already_seen(self, message_id: str) -> bool:
        """Return True if this Message-ID is already in the database."""
        if not message_id:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM classifications WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            return row is not None

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent `limit` classifications, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM classifications ORDER BY classified_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> Dict[str, int]:
        """Return aggregate counts: total, phishing, benign."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM classifications"
            ).fetchone()[0]
            phishing = conn.execute(
                "SELECT COUNT(*) FROM classifications WHERE prediction = 'phishing'"
            ).fetchone()[0]
        return {"total": total, "phishing": phishing, "benign": total - phishing}
