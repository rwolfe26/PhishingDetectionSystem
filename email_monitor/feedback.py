"""
Feedback Store

Persists user corrections to classification results.
Each record stores the raw email text, what the model predicted,
what the user says it should be, and whether the model was right.

Used for active learning: run_pipeline.py --include-feedback will
pull all corrections into the next training run.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


class FeedbackStore:
    """
    SQLite-backed store for user feedback on classification results.

    Corrections (where predicted != correct_label) are the training
    signal — they represent emails the model got wrong and should
    learn from on the next retrain.
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS feedback (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at   TEXT    NOT NULL,
            email_hash     TEXT    NOT NULL,
            email_text     TEXT    NOT NULL,
            predicted      TEXT    NOT NULL,
            correct_label  TEXT    NOT NULL,
            confidence     REAL,
            was_correct    INTEGER NOT NULL   -- 1 = model was right, 0 = correction
        );
        CREATE INDEX IF NOT EXISTS idx_fb_correct   ON feedback (was_correct);
        CREATE INDEX IF NOT EXISTS idx_fb_submitted ON feedback (submitted_at);
    """

    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._DDL)

    # ── Write ────────────────────────────────────────────────────────────────

    def record(
        self,
        email_text: str,
        predicted: str,
        correct_label: str,
        confidence: float = 0.0,
    ) -> int:
        """
        Store a feedback record.

        Args:
            email_text:    Full raw email text.
            predicted:     What the model said ("phishing" | "benign").
            correct_label: What the user says it should be.
            confidence:    Model's confidence at classification time.

        Returns:
            rowid of the inserted row.
        """
        email_hash = hashlib.sha256(email_text.encode()).hexdigest()
        was_correct = 1 if predicted == correct_label else 0
        cur = self._conn.execute(
            """
            INSERT INTO feedback
                (submitted_at, email_hash, email_text, predicted,
                 correct_label, confidence, was_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                email_hash,
                email_text,
                predicted,
                correct_label,
                float(confidence),
                was_correct,
            ),
        )
        self._conn.commit()
        return cur.lastrowid or 0

    # ── Read ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return aggregate feedback counts."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]
        corrections = self._conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE was_correct = 0"
        ).fetchone()[0]
        confirmations = total - corrections
        return {
            "total": total,
            "corrections": corrections,
            "confirmations": confirmations,
        }

    def corrections(self) -> List[Tuple[str, int]]:
        """
        Return all corrections as (email_text, label_int) tuples,
        ready to merge into a training dataset.

        label_int: 1 = phishing, 0 = benign
        """
        rows = self._conn.execute(
            "SELECT email_text, correct_label FROM feedback WHERE was_correct = 0"
        ).fetchall()
        return [
            (row["email_text"], 1 if row["correct_label"] == "phishing" else 0)
            for row in rows
        ]

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent feedback records, newest first."""
        rows = self._conn.execute(
            """SELECT id, submitted_at, predicted, correct_label,
                      confidence, was_correct
               FROM feedback
               ORDER BY submitted_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
