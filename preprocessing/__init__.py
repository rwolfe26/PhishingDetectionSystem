"""
Email Preprocessing Pipeline

Unified interface for parsing raw emails and extracting features for ML models.

Usage:
    from preprocessing import preprocess_email, EmailFeatures

    result = preprocess_email(raw_email_text)
    print(result['subject'])
    print(result['features'].to_dict())

    # With LSA embeddings:
    from preprocessing import preprocess_email_with_lsa, fit_lsa_encoder

    # First, fit the LSA encoder on your corpus
    lsa_encoder = fit_lsa_encoder(training_emails, n_components=768)

    # Then use it in preprocessing
    result = preprocess_email_with_lsa(raw_email_text, lsa_encoder)
    print(result['lsa_embedding'].shape)  # (768,)
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

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

# Conditional import for LSA to avoid hard dependency
if TYPE_CHECKING:
    from bert_base.lsa_tool import LSATextEncoder


__all__ = [
    # Main functions
    'preprocess_email',
    'preprocess_email_with_lsa',
    'preprocess_email_batch_with_lsa',
    'fit_lsa_encoder',
    'fit_lsa_encoder_from_texts',
    'fit_and_extract_features',
    'get_combined_feature_vector',

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


def _get_lsa_encoder(n_components, max_features, min_df, max_df):
    """Import and instantiate LSATextEncoder."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'bert_base'))
    from lsa_tool import LSATextEncoder
    return LSATextEncoder(
        n_components=n_components,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )


def fit_lsa_encoder(raw_emails: List[str],
                    n_components: int = 768,
                    max_features: Optional[int] = None,
                    min_df: int = 1,
                    max_df: float = 0.85) -> 'LSATextEncoder':
    """
    Fit an LSA encoder on a corpus of emails.

    Preprocesses emails to extract body text, then trains an LSA model.
    For efficiency, prefer `fit_lsa_encoder_from_texts` if you already have
    preprocessed texts, or use `fit_and_extract_features` to do everything
    in one pass.

    Args:
        raw_emails: List of raw email texts to train on
        n_components: Number of LSA dimensions (default: 768)
        max_features: Optional limit on vocabulary size
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency (proportion)

    Returns:
        Fitted LSATextEncoder instance
    """
    print(f"Preprocessing {len(raw_emails)} emails for LSA training...")
    body_texts = []
    for i, email in enumerate(raw_emails):
        result = preprocess_email(email)
        body_texts.append(f"{result['subject']} {result['body_text']}")
        if (i + 1) % 1000 == 0:
            print(f"  Preprocessed {i+1}/{len(raw_emails)}...")

    return fit_lsa_encoder_from_texts(body_texts, n_components, max_features, min_df, max_df)


def fit_lsa_encoder_from_texts(body_texts: List[str],
                                n_components: int = 768,
                                max_features: Optional[int] = None,
                                min_df: int = 1,
                                max_df: float = 0.85) -> 'LSATextEncoder':
    """
    Fit an LSA encoder on pre-extracted body texts (skips email parsing).

    Args:
        body_texts: List of pre-extracted text strings (subject + body)
        n_components: Number of LSA dimensions
        max_features: Optional vocab cap
        min_df: Minimum document frequency
        max_df: Maximum document frequency

    Returns:
        Fitted LSATextEncoder instance
    """
    encoder = _get_lsa_encoder(n_components, max_features, min_df, max_df)

    print(f"Fitting LSA encoder with {n_components} components on {len(body_texts)} texts...")
    encoder.fit(body_texts)

    explained_var = encoder.explained_variance_ratio()
    if explained_var is not None:
        print(f"LSA encoder fitted. Total explained variance: {explained_var.sum():.3f}")

    return encoder


def fit_and_extract_features(raw_emails: List[str],
                              lsa_encoder: 'LSATextEncoder') -> tuple:
    """
    Single-pass preprocessing: parse emails once, return feature matrix and texts.

    This avoids the double-preprocessing that occurs when calling fit_lsa_encoder
    followed by preprocess_email_batch_with_lsa separately.

    Args:
        raw_emails: List of raw email texts
        lsa_encoder: Pre-fitted LSATextEncoder

    Returns:
        Tuple of (feature_matrix: np.ndarray, body_texts: List[str])
        where feature_matrix has shape (n_emails, n_numeric + n_lsa)
    """
    results = []
    body_texts = []

    for i, email in enumerate(raw_emails):
        result = preprocess_email(email)
        body_texts.append(f"{result['subject']} {result['body_text']}")
        results.append(result)
        if (i + 1) % 1000 == 0:
            print(f"  Preprocessed {i+1}/{len(raw_emails)}...")

    # Batch transform with LSA (much faster than one-at-a-time)
    lsa_embeddings = lsa_encoder.transform(body_texts)

    feature_matrix = np.array([
        np.concatenate([
            np.array(r['feature_vector'], dtype=np.float32),
            lsa_embeddings[i]
        ])
        for i, r in enumerate(results)
    ])

    return feature_matrix, body_texts


def preprocess_email_with_lsa(raw_email: str,
                               lsa_encoder: 'LSATextEncoder') -> Dict[str, Any]:
    """
    Complete preprocessing pipeline including LSA embeddings.

    This function runs the standard preprocessing (parse → URLs → features)
    and additionally generates a 768-dimensional semantic embedding using
    the provided LSA encoder.

    Args:
        raw_email: Raw email text (RFC 822 / mbox format)
        lsa_encoder: Pre-fitted LSATextEncoder instance

    Returns:
        Dictionary containing all standard preprocessing outputs PLUS:
        - lsa_embedding: numpy array of shape (768,) with semantic embedding
        - combined_vector: numpy array of shape (802,) = 34 features + 768 LSA

    Example:
        >>> encoder = fit_lsa_encoder(training_emails)
        >>> result = preprocess_email_with_lsa(test_email, encoder)
        >>> print(result['lsa_embedding'].shape)  # (768,)
        >>> print(result['combined_vector'].shape)  # (802,)
    """
    # Run standard preprocessing
    result = preprocess_email(raw_email)

    # Generate LSA embedding from subject + body
    combined_text = f"{result['subject']} {result['body_text']}"
    lsa_embedding = lsa_encoder.transform([combined_text])[0]  # Shape: (768,)

    # Add LSA embedding to result
    result['lsa_embedding'] = lsa_embedding

    # Create combined feature vector: [34 numeric features] + [768 LSA dims]
    numeric_features = np.array(result['feature_vector'], dtype=np.float32)
    combined = np.concatenate([numeric_features, lsa_embedding])
    result['combined_vector'] = combined

    return result


def preprocess_email_batch_with_lsa(raw_emails: List[str],
                                     lsa_encoder: 'LSATextEncoder') -> List[Dict[str, Any]]:
    """
    Process multiple emails with LSA embeddings.

    Args:
        raw_emails: List of raw email texts
        lsa_encoder: Pre-fitted LSATextEncoder instance

    Returns:
        List of preprocessing results, each including LSA embeddings

    Example:
        >>> encoder = fit_lsa_encoder(training_emails)
        >>> results = preprocess_email_batch_with_lsa(test_emails, encoder)
        >>> embeddings = [r['lsa_embedding'] for r in results]
    """
    return [preprocess_email_with_lsa(email, lsa_encoder) for email in raw_emails]


def get_combined_feature_vector(raw_email: str,
                                lsa_encoder: 'LSATextEncoder') -> np.ndarray:
    """
    Extract the complete 802-dimensional feature vector for ML models.

    This is a convenience function that returns just the combined feature
    vector (34 numeric features + 768 LSA dimensions) suitable for direct
    input to classifiers.

    Args:
        raw_email: Raw email text
        lsa_encoder: Pre-fitted LSATextEncoder instance

    Returns:
        numpy array of shape (802,) containing:
        - [0:34]: Numeric features (URLs, text, headers, HTML, attachments)
        - [34:802]: LSA semantic embedding

    Example:
        >>> encoder = fit_lsa_encoder(training_emails)
        >>> X = np.array([get_combined_feature_vector(email, encoder)
        ...               for email in test_emails])
        >>> print(X.shape)  # (n_emails, 802)
        >>> # Now X is ready for your classifier
        >>> predictions = model.predict(X)
    """
    result = preprocess_email_with_lsa(raw_email, lsa_encoder)
    return result['combined_vector']
