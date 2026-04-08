"""
Explainability Module

Provides per-prediction explanations using:
1. Random Forest feature importances weighted by feature values
2. Optional SHAP TreeExplainer (if shap is installed)
3. Phishing indicator highlighting from the feature extractor
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from pipeline.trainer import Trainer

# Feature names: 42 numeric + LSA dims
NUMERIC_NAMES = Trainer.NUMERIC_FEATURE_NAMES
N_NUMERIC = len(NUMERIC_NAMES)

# Human-readable descriptions for each numeric feature
FEATURE_DESCRIPTIONS = {
    'num_urls': 'Number of URLs in email',
    'num_unique_domains': 'Number of unique domains',
    'has_ip_url': 'URL using raw IP address',
    'no_https_ratio': 'Fraction of URLs without HTTPS',
    'avg_url_length': 'Average URL length',
    'max_url_length': 'Longest URL length',
    'avg_path_depth': 'Average URL path depth',
    'total_dots_in_urls': 'Total dots in all URL domains',
    'has_at_symbol_url': 'URL contains @ symbol',
    'has_suspicious_port': 'URL uses suspicious port',
    'num_words': 'Word count in body',
    'num_unique_words': 'Unique word count',
    'num_chars': 'Character count',
    'num_special_chars': 'Special character count',
    'num_urgent_keywords': 'Urgency keyword matches',
    'num_credential_keywords': 'Credential request keywords',
    'num_action_keywords': 'Action prompt keywords',
    'has_reply_to_mismatch': 'Reply-To domain differs from From',
    'has_return_path_mismatch': 'Return-Path domain differs from From',
    'num_received_hops': 'Email relay hops',
    'has_suspicious_mailer': 'Known phishing mailer detected',
    'subject_has_urgent': 'Urgency words in subject',
    'subject_has_re_fw': 'Fake Re:/Fw: in subject',
    'has_html': 'Email contains HTML',
    'has_form': 'HTML form detected',
    'has_iframe': 'iframe detected in HTML',
    'has_hidden_text': 'Hidden text in HTML',
    'num_external_links': 'External HTTP/HTTPS links',
    'link_text_url_mismatch': 'Link text URL differs from href',
    'num_attachments': 'Number of attachments',
    'has_executable_attachment': 'Executable file attachment',
    'has_archive_attachment': 'Archive file attachment',
    'spf_dkim_fail': 'SPF/DKIM authentication failed',
    'sender_domain_mismatch': 'Sender display name impersonation',
    'num_homograph_chars': 'IDN/homograph characters in URLs',
    'brand_impersonation_score': 'Brand impersonation similarity score',
    'urgency_density': 'Urgency keyword density',
    'html_text_ratio': 'HTML-to-text length ratio',
    'num_shortener_urls': 'URL shortener links detected',
    'greeting_generic': 'Generic greeting ("Dear Customer")',
    'num_auth_keywords': 'Authentication action keywords',
    'subject_all_caps_ratio': 'Fraction of ALL CAPS words in subject',
    'has_redirect_url': 'Shortener URL resolved to a different final domain',
    'num_redirect_hops': 'Total redirect hops across resolved shortener URLs',
}


def explain_prediction(pipeline, feature_vector: np.ndarray,
                        top_n: int = 8) -> List[Dict]:
    """
    Produce a ranked list of the top-N features driving the prediction.

    Uses RF feature importances as a proxy for local explanation.

    Args:
        pipeline: Trained EmailPhishingPipeline
        feature_vector: Combined feature vector (42 numeric + LSA dims)
        top_n: Number of top features to return

    Returns:
        List of dicts with keys: name, description, value, importance, direction
    """
    clf = pipeline.classifier
    if clf is None or not hasattr(clf, 'feature_importances_'):
        return []

    importances = clf.feature_importances_

    # Only explain numeric features (LSA dims are not human-interpretable individually)
    numeric_importances = importances[:N_NUMERIC]
    numeric_values = feature_vector[:N_NUMERIC]

    # Weight by |feature value| to approximate local importance
    local_weights = numeric_importances * (np.abs(numeric_values) + 1e-8)
    top_indices = np.argsort(local_weights)[-top_n:][::-1]

    explanations = []
    for idx in top_indices:
        name = NUMERIC_NAMES[idx] if idx < N_NUMERIC else f'lsa_{idx - N_NUMERIC}'
        value = float(numeric_values[idx])
        importance = float(numeric_importances[idx])

        if importance < 1e-6:
            continue

        explanations.append({
            'name': name,
            'description': FEATURE_DESCRIPTIONS.get(name, name),
            'value': round(value, 4),
            'importance': round(importance, 6),
            'direction': 'phishing' if value > 0 else 'benign',
        })

    return explanations


def explain_with_shap(pipeline, feature_vector: np.ndarray,
                      top_n: int = 8) -> Optional[List[Dict]]:
    """
    Use SHAP TreeExplainer for local explanations if shap is available.

    Args:
        pipeline: Trained EmailPhishingPipeline
        feature_vector: Combined feature vector
        top_n: Number of top features to return

    Returns:
        List of explanation dicts, or None if shap is unavailable
    """
    try:
        import shap
    except ImportError:
        return None

    clf = pipeline.classifier
    if clf is None:
        return None

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(feature_vector.reshape(1, -1))

        # For binary classification: use SHAP values for the positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1][0]  # class 1 = phishing
        else:
            sv = shap_values[0]

        numeric_sv = sv[:N_NUMERIC]
        top_indices = np.argsort(np.abs(numeric_sv))[-top_n:][::-1]

        explanations = []
        for idx in top_indices:
            name = NUMERIC_NAMES[idx] if idx < N_NUMERIC else f'lsa_{idx - N_NUMERIC}'
            shap_val = float(numeric_sv[idx])
            feature_val = float(feature_vector[idx])

            explanations.append({
                'name': name,
                'description': FEATURE_DESCRIPTIONS.get(name, name),
                'value': round(feature_val, 4),
                'shap_value': round(shap_val, 6),
                'direction': 'phishing' if shap_val > 0 else 'benign',
            })

        return explanations
    except Exception:
        return None


def highlight_phishing_indicators(email_text: str,
                                   feature_vector: np.ndarray) -> Dict:
    """
    Extract textual phishing indicators found in the email for UI display.

    Args:
        email_text: Raw email text
        feature_vector: Feature vector (to pull computed values from)

    Returns:
        Dict with lists of detected indicators
    """
    from preprocessing.feature_extractor import FeatureExtractor
    import re

    text_lower = email_text.lower()
    indicators = {
        'urgent_phrases': [],
        'credential_phrases': [],
        'action_phrases': [],
        'suspicious_urls': [],
        'auth_phrases': [],
    }

    for kw in FeatureExtractor.URGENT_KEYWORDS:
        if kw in text_lower:
            indicators['urgent_phrases'].append(kw)

    for kw in FeatureExtractor.CREDENTIAL_KEYWORDS:
        if kw in text_lower:
            indicators['credential_phrases'].append(kw)

    for kw in FeatureExtractor.ACTION_KEYWORDS:
        if kw in text_lower:
            indicators['action_phrases'].append(kw)

    for kw in FeatureExtractor.AUTH_KEYWORDS:
        if kw in text_lower:
            indicators['auth_phrases'].append(kw)

    # Flag URL shorteners found in text
    url_pattern = re.compile(r'https?://[\w./%-]+', re.IGNORECASE)
    for url in url_pattern.findall(email_text):
        domain = re.sub(r'https?://', '', url).split('/')[0].lower()
        if domain in FeatureExtractor.URL_SHORTENERS:
            indicators['suspicious_urls'].append(url)

    return indicators
