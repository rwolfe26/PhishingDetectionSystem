"""
Email Preprocessing Pipeline

Unified interface for parsing raw emails and extracting features for ML models.

Usage:
    from preprocessing import preprocess_email, EmailFeatures
    
    result = preprocess_email(raw_email_text)
    print(result['subject'])
    print(result['features'].to_dict())
"""

from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .email_parser import (
    EmailParser,
    ParsedEmail,
    AttachmentInfo,
    parse_email,
)

from .url_extractor import (
    URLExtractor,
    URLInfo,
    extract_urls,
    get_unique_domains,
)

from .feature_extractor import (
    FeatureExtractor,
    EmailFeatures,
    extract_features,
)


__all__ = [
    # Main function
    'preprocess_email',
    
    # Parser classes
    'EmailParser',
    'ParsedEmail',
    'AttachmentInfo',
    'parse_email',
    
    # URL extractor
    'URLExtractor',
    'URLInfo',
    'extract_urls',
    'get_unique_domains',
    
    # Feature extractor
    'FeatureExtractor',
    'EmailFeatures',
    'extract_features',
]


def preprocess_email(raw_email: str) -> Dict[str, Any]:
    """
    Complete email preprocessing pipeline.
    
    Chains: raw text → parse → URL extraction → feature extraction
    
    Args:
        raw_email: Raw email text (RFC 822 / mbox format)
        
    Returns:
        Dictionary containing:
        - parsed: ParsedEmail object
        - subject: Email subject string
        - body_text: Plain text body
        - body_html: HTML body (if present)
        - headers: Dict of all headers
        - from_address: Sender address
        - urls: List of URLInfo objects
        - unique_domains: Set of unique domains from URLs
        - attachments: List of AttachmentInfo objects
        - features: EmailFeatures object with numeric features
        - feature_dict: Dict of feature name → value
        - feature_vector: List of feature values (ordered)
    """
    # Step 1: Parse email
    parser = EmailParser()
    parsed = parser.parse(raw_email)
    
    # Step 2: Extract URLs
    url_extractor = URLExtractor()
    urls = url_extractor.extract_all(
        text=parsed.body_text,
        html=parsed.body_html
    )
    unique_domains = get_unique_domains(urls)
    
    # Step 3: Extract features
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract(parsed, urls)
    
    # Build result dictionary
    return {
        # Parsed components
        'parsed': parsed,
        'subject': parsed.subject,
        'body_text': parsed.body_text,
        'body_html': parsed.body_html,
        'headers': parsed.headers,
        'from_address': parsed.from_address,
        
        # URLs
        'urls': urls,
        'unique_domains': unique_domains,
        
        # Attachments
        'attachments': parsed.attachments,
        
        # Features for ML
        'features': features,
        'feature_dict': features.to_dict(),
        'feature_vector': features.to_list(),
        'feature_names': EmailFeatures.feature_names(),
    }


def preprocess_email_batch(raw_emails: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple emails.
    
    Args:
        raw_emails: List of raw email texts
        
    Returns:
        List of preprocessing results
    """
    return [preprocess_email(email) for email in raw_emails]


def get_feature_matrix(raw_emails: List[str]) -> tuple:
    """
    Extract feature matrix from multiple emails.
    
    Args:
        raw_emails: List of raw email texts
        
    Returns:
        Tuple of (feature_matrix, feature_names) where feature_matrix
        is a list of feature vectors suitable for ML models
    """
    results = preprocess_email_batch(raw_emails)
    feature_matrix = [r['feature_vector'] for r in results]
    feature_names = EmailFeatures.feature_names()
    return feature_matrix, feature_names
