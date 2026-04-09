#!/usr/bin/env python3
"""
Email Monitor CLI — Phase 2

Poll an IMAP mailbox and classify every incoming email for phishing.
Results are stored in a local SQLite database and printed to the terminal
with colour-coded alerts.

━━━  Quick start  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Continuous monitoring (checks every 60 s)
  python monitor.py --host imap.gmail.com --user you@gmail.com \\
                    --password YOUR_APP_PASSWORD

  # Single scan then exit
  python monitor.py --host imap.gmail.com --user you@gmail.com \\
                    --password YOUR_APP_PASSWORD --scan-once

  # Pass password via environment variable (safer — avoids shell history)
  IMAP_PASSWORD=YOUR_APP_PASSWORD python monitor.py \\
      --host imap.gmail.com --user you@gmail.com

━━━  Gmail app-password setup  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Enable 2-Step Verification on your Google account.
  2. Go to  myaccount.google.com/apppasswords
  3. Create a new app password → Mail → Other (name it "phishing-monitor").
  4. Use the generated 16-character password as --password.
     (Never use your regular Google password here.)

━━━  Other providers  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Outlook / Hotmail:  --host imap-mail.outlook.com
  Yahoo Mail:         --host imap.mail.yahoo.com
  iCloud Mail:        --host imap.mail.me.com
  Any IMAP provider:  --host mail.yourprovider.com --port 993
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Make sure the project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.core import EmailPhishingPipeline
from email_monitor import EmailMonitor
from email_monitor.notifier import print_status, print_warning


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description="Monitor an IMAP mailbox for phishing emails.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Connection ────────────────────────────────────────────────────────────
    conn = parser.add_argument_group("IMAP connection")
    conn.add_argument(
        "--host", required=True,
        help="IMAP server hostname  (e.g. imap.gmail.com)",
    )
    conn.add_argument(
        "--user", required=True,
        help="Your email address / IMAP username",
    )
    conn.add_argument(
        "--password",
        help="IMAP / app password.  Alternatively set IMAP_PASSWORD env var.",
    )
    conn.add_argument(
        "--port", type=int, default=993,
        help="IMAP SSL port  (default: 993)",
    )
    conn.add_argument(
        "--mailbox", default="INBOX",
        help="Mailbox / folder to monitor  (default: INBOX)",
    )

    # ── Monitor behaviour ─────────────────────────────────────────────────────
    mon = parser.add_argument_group("Monitor behaviour")
    mon.add_argument(
        "--interval", type=int, default=60,
        help="Seconds between polls in continuous mode  (default: 60)",
    )
    mon.add_argument(
        "--scan-once", action="store_true",
        help="Classify unseen emails once and exit instead of looping",
    )
    mon.add_argument(
        "--db", default="monitor.db",
        help="SQLite database file for classification history  (default: monitor.db)",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    mod = parser.add_argument_group("Model")
    mod.add_argument(
        "--model-dir", default="./models",
        help="Directory containing trained .pkl model files  (default: ./models)",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Resolve password ──────────────────────────────────────────────────────
    password = args.password or os.environ.get("IMAP_PASSWORD", "")
    if not password:
        parser.error(
            "Password required.  Use --password or set the IMAP_PASSWORD "
            "environment variable.\n"
            "For Gmail, create an App Password at "
            "myaccount.google.com/apppasswords"
        )

    # ── Load the trained pipeline ─────────────────────────────────────────────
    model_dir = args.model_dir
    if not Path(model_dir).exists():
        print_warning(
            f"Model directory '{model_dir}' not found.  "
            "Run:  python run_pipeline.py --train"
        )
        sys.exit(1)

    print_status(f"Loading model from {model_dir} ...")
    try:
        pipeline = EmailPhishingPipeline.load(model_dir)
    except Exception as exc:
        print_warning(f"Failed to load model: {exc}")
        sys.exit(1)
    print_status("Model loaded.\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    monitor = EmailMonitor(
        pipeline=pipeline,
        host=args.host,
        user=args.user,
        password=password,
        port=args.port,
        mailbox=args.mailbox,
        db_path=args.db,
    )

    if args.scan_once:
        count = monitor.scan_once()
        print_status(f"Done — {count} email(s) newly classified.")
        print_status(f"History saved to: {args.db}")
    else:
        monitor.run(interval=args.interval)


if __name__ == "__main__":
    main()
