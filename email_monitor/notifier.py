"""
Terminal Notifier

Colour-coded stdout alerts for phishing detections and status messages.
No external dependencies — uses ANSI escape codes only.
"""

import sys
from typing import List

# ANSI codes — only emitted when stdout is a real terminal
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _c(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


def alert_phishing(
    subject: str,
    sender: str,
    confidence: float,
    risk_level: str,
    top_features: List[dict],
) -> None:
    """Print a prominent phishing alert."""
    border = "=" * 62
    print(f"\n{_c(_RED + _BOLD, border)}")
    print(_c(_RED + _BOLD, f"  ⚠  PHISHING DETECTED — {risk_level} RISK"))
    print(_c(_RED + _BOLD, border))
    print(f"  {_c(_BOLD, 'From:')}       {sender}")
    print(f"  {_c(_BOLD, 'Subject:')}    {subject}")
    print(f"  {_c(_BOLD, 'Confidence:')} {confidence * 100:.1f}%")
    if top_features:
        names = [f.get("feature", "") for f in top_features[:3] if f.get("feature")]
        if names:
            print(f"  {_c(_BOLD, 'Top signals:')} {', '.join(names)}")
    print(_c(_RED + _BOLD, border) + "\n")


def alert_benign(subject: str, sender: str, confidence: float) -> None:
    """Print a quiet one-liner for safe emails."""
    label = _c(_GREEN, "✓ SAFE")
    pct = f"{(1 - confidence) * 100:.0f}%" if confidence < 0.5 else f"{confidence * 100:.0f}%"
    print(f"  {label}  [{pct:>4}]  {sender[:42]:<42}  {subject[:50]}")


def print_status(message: str) -> None:
    """Print a cyan status/info line."""
    print(_c(_CYAN, f"[monitor] {message}"))


def print_warning(message: str) -> None:
    """Print a yellow warning line."""
    print(_c(_YELLOW, f"[monitor] WARNING: {message}"), file=sys.stderr)


def print_summary(stats: dict) -> None:
    """Print end-of-session summary."""
    print(f"\n{_c(_BOLD, '─' * 40)}")
    print(_c(_BOLD, "Session summary"))
    print(f"  Scanned:  {stats['total']}")
    print(f"  {_c(_RED, 'Phishing:')} {stats['phishing']}")
    print(f"  {_c(_GREEN, 'Benign:')}   {stats['benign']}")
    print(_c(_BOLD, "─" * 40))
