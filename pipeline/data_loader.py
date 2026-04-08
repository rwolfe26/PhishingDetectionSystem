"""
Data Loading Utilities

Handles loading email datasets from directories and organizing them for training.
Supports raw email files, CSV, and JSONL phishing datasets.
"""

import csv
import json
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np


class DataLoader:
    """Utility class for loading email datasets."""

    @staticmethod
    def load_emails_from_directory(directory: Path, label: int) -> Tuple[List[str], List[int]]:
        """
        Load emails from a directory and assign labels.

        Args:
            directory: Path to directory containing email files
            label: Label to assign (0=ham, 1=spam/phishing)

        Returns:
            Tuple of (emails, labels)
        """
        emails = []
        labels = []

        if not directory.exists():
            print(f"Warning: Directory not found: {directory}")
            return emails, labels

        for filename in os.listdir(directory):
            filepath = directory / filename
            if filepath.is_file():
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        email_content = f.read()
                        emails.append(email_content)
                        labels.append(label)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        return emails, labels

    @staticmethod
    def load_phishing_csv(csv_path: Path, max_samples: int = None) -> Tuple[List[str], List[int]]:
        """
        Load emails from Phishing_Email.csv (Email Text + Email Type columns).

        Labels: 'Phishing Email' → 1, 'Safe Email' → 0

        Args:
            csv_path: Path to the CSV file
            max_samples: Maximum number of rows to load per class (None = all)

        Returns:
            Tuple of (email_texts, labels)
        """
        emails = []
        labels = []
        ham_count = 0
        phish_count = 0

        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get('Email Text', '').strip()
                    email_type = row.get('Email Type', '').strip().lower()

                    if not text:
                        continue

                    if 'phishing' in email_type:
                        if max_samples and phish_count >= max_samples:
                            continue
                        emails.append(text)
                        labels.append(1)
                        phish_count += 1
                    elif 'safe' in email_type:
                        if max_samples and ham_count >= max_samples:
                            continue
                        emails.append(text)
                        labels.append(0)
                        ham_count += 1

                    if max_samples and ham_count >= max_samples and phish_count >= max_samples:
                        break

        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

        print(f"  Loaded from CSV: {ham_count} safe, {phish_count} phishing")
        return emails, labels

    @staticmethod
    def load_jsonl_phishing(jsonl_path: Path) -> Tuple[List[str], List[int]]:
        """
        Load emails from a JSONL file (fields: subject, body, label).

        Labels: 'phishing' → 1, 'benign'/'safe' → 0

        Args:
            jsonl_path: Path to the JSONL file

        Returns:
            Tuple of (email_texts, labels)
        """
        emails = []
        labels = []

        try:
            with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    subject = record.get('subject', '')
                    body = record.get('body', '')
                    text = f"Subject: {subject}\n\n{body}".strip()
                    raw_label = record.get('label', '').lower()

                    if not text:
                        continue

                    if 'phishing' in raw_label:
                        emails.append(text)
                        labels.append(1)
                    elif raw_label in ('benign', 'safe', 'ham'):
                        emails.append(text)
                        labels.append(0)

        except Exception as e:
            print(f"Error reading {jsonl_path}: {e}")

        phish = sum(1 for l in labels if l == 1)
        ham = sum(1 for l in labels if l == 0)
        print(f"  Loaded from JSONL: {ham} benign, {phish} phishing")
        return emails, labels

    @staticmethod
    def load_dataset(ham_dirs: List[Path], spam_dirs: List[Path]) -> Tuple[List[str], np.ndarray]:
        """
        Load complete dataset from multiple directories.

        Args:
            ham_dirs: List of directories containing ham emails
            spam_dirs: List of directories containing spam/phishing emails

        Returns:
            Tuple of (emails, labels)
        """
        all_emails = []
        all_labels = []

        for ham_dir in ham_dirs:
            print(f"Loading ham from {ham_dir.name}...")
            emails, labels = DataLoader.load_emails_from_directory(ham_dir, label=0)
            all_emails.extend(emails)
            all_labels.extend(labels)
            print(f"  {len(emails)} ham emails")

        for spam_dir in spam_dirs:
            print(f"Loading spam from {spam_dir.name}...")
            emails, labels = DataLoader.load_emails_from_directory(spam_dir, label=1)
            all_emails.extend(emails)
            all_labels.extend(labels)
            print(f"  {len(emails)} spam emails")

        print(f"\nDirectory totals: {len(all_emails)} emails")
        print(f"  Ham: {sum(1 for l in all_labels if l == 0)}")
        print(f"  Spam: {sum(1 for l in all_labels if l == 1)}")

        return all_emails, np.array(all_labels)
