"""
Email Phishing Detection Pipeline Package

A modular pipeline for training and deploying phishing email classifiers
using preprocessing and LSA semantic analysis.
"""

from .core import EmailPhishingPipeline
from .data_loader import DataLoader
from .trainer import Trainer
from .predictor import Predictor

__all__ = [
    'EmailPhishingPipeline',
    'DataLoader',
    'Trainer',
    'Predictor',
]

__version__ = '1.0.0'
