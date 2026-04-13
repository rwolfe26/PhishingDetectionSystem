"""
Data Loading Utilities

Handles loading email datasets from directories and organizing them for training.
"""

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
            label: Label to assign (0=ham, 1=spam)

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
    def load_dataset(ham_dirs: List[Path], spam_dirs: List[Path]) -> Tuple[List[str], np.ndarray]:
        """
        Load complete dataset from multiple directories.

        Args:
            ham_dirs: List of directories containing ham emails
            spam_dirs: List of directories containing spam emails

        Returns:
            Tuple of (emails, labels)
        """
        all_emails = []
        all_labels = []

        # Load ham emails (label=0)
        for ham_dir in ham_dirs:
            print(f"Loading ham emails from {ham_dir}...")
            emails, labels = DataLoader.load_emails_from_directory(ham_dir, label=0)
            all_emails.extend(emails)
            all_labels.extend(labels)
            print(f"  Loaded {len(emails)} ham emails")

        # Load spam emails (label=1)
        for spam_dir in spam_dirs:
            print(f"Loading spam emails from {spam_dir}...")
            emails, labels = DataLoader.load_emails_from_directory(spam_dir, label=1)
            all_emails.extend(emails)
            all_labels.extend(labels)
            print(f"  Loaded {len(emails)} spam emails")

        print(f"\nTotal dataset: {len(all_emails)} emails")
        print(f"  Ham: {sum(1 for l in all_labels if l == 0)}")
        print(f"  Spam: {sum(1 for l in all_labels if l == 1)}")

        return all_emails, np.array(all_labels)
