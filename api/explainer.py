"""
Explainability Module

Provides per-prediction explanations using:
1. Random Forest feature importances weighted by feature values
2. Optional SHAP TreeExplainer (if shap is installed)
3. Phishing indicator highlighting from the feature extractor
"""

from typing import Dict, List, Optional
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
    'special_char_ratio': 'Special chars as fraction of total text',
    'unique_word_ratio': 'Lexical diversity (unique / total words)',
    'caps_ratio': 'Uppercase letter ratio',
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


def generate_plain_english_summary(
    prediction: str,
    confidence: float,
    risk_level: str,
    top_features: list,
    indicators: dict,
    feature_vector=None,
) -> str:
    """
    Generate a plain-English explanation of the classification for non-technical users.

    Reads from the full feature vector (all 44 numeric features) when available so it
    can reference any signal, not just the top-8 returned to the UI.
    """
    # Build a name→value map from the full feature vector, falling back to top_features
    fv: dict = {}
    if feature_vector is not None:
        for i, name in enumerate(NUMERIC_NAMES):
            if i < len(feature_vector):
                fv[name] = float(feature_vector[i])
    for f in top_features:
        fv.setdefault(f['name'], f['value'])

    def fval(name: str) -> float:
        return fv.get(name, 0.0)

    urgent_phrases     = indicators.get('urgent_phrases', [])
    credential_phrases = indicators.get('credential_phrases', [])
    suspicious_urls    = indicators.get('suspicious_urls', [])

    # ── Phishing branch ───────────────────────────────────────────────────────
    if prediction == 'phishing':
        pct = int(round(confidence * 100))
        if risk_level == 'HIGH':
            opener = f"This email is very likely a phishing attempt ({pct}% confidence)."
        elif risk_level == 'MEDIUM':
            opener = f"This email shows multiple signs of phishing ({pct}% confidence)."
        else:
            opener = (
                f"This email has some suspicious characteristics, "
                f"though the signal is weaker ({pct}% confidence)."
            )

        reasons = []

        if urgent_phrases:
            sample = ', '.join(f'"{p}"' for p in urgent_phrases[:2])
            tail   = f" and {len(urgent_phrases) - 2} more" if len(urgent_phrases) > 2 else ""
            reasons.append(
                f"it uses urgency phrases like {sample}{tail} "
                f"designed to make you act without thinking"
            )

        if credential_phrases:
            reasons.append(
                "it asks you to submit sensitive information "
                "such as passwords or account credentials"
            )

        if fval('sender_domain_mismatch') > 0:
            reasons.append(
                "the sender's display name doesn't match their actual email address, "
                "a classic impersonation tactic"
            )
        elif fval('has_reply_to_mismatch') > 0:
            reasons.append(
                "the Reply-To address differs from the sender's address, "
                "used to redirect your replies to the attacker"
            )

        if fval('brand_impersonation_score') > 0.3:
            reasons.append(
                "the wording and structure closely imitate a well-known brand or organisation"
            )

        url_shortener_count = int(fval('num_shortener_urls'))
        if suspicious_urls or url_shortener_count > 0:
            n         = max(len(suspicious_urls), url_shortener_count)
            link_word = "a link" if n == 1 else f"{n} links"
            reasons.append(f"{link_word} use URL shorteners that conceal the real destination")

        if fval('has_ip_url') > 0:
            reasons.append(
                "at least one link uses a raw IP address instead of a normal domain, "
                "which legitimate services never do"
            )

        if fval('has_redirect_url') > 0:
            reasons.append(
                "at least one link redirects through multiple websites before reaching "
                "its final destination, a technique used to evade detection"
            )

        if fval('spf_dkim_fail') > 0:
            reasons.append(
                "the email failed standard authentication checks (SPF/DKIM), "
                "indicating it may not genuinely come from the claimed sender"
            )

        if fval('greeting_generic') > 0:
            reasons.append(
                'it uses a generic greeting like "Dear Customer" instead of your name, '
                "typical of mass phishing campaigns"
            )

        if fval('has_executable_attachment') > 0:
            reasons.append(
                "it contains an executable file attachment, "
                "which legitimate services almost never send via email"
            )

        if fval('subject_all_caps_ratio') > 0.5:
            reasons.append(
                "the subject line is heavily capitalised to manufacture a false sense of urgency"
            )

        if not reasons:
            return (
                f"{opener} The model's analysis of language patterns, structure, and "
                "technical signals in this email strongly suggest phishing intent, "
                "even without obvious keyword red flags."
            )

        if len(reasons) == 1:
            body = f"The key red flag is that {reasons[0]}."
        elif len(reasons) == 2:
            body = f"The main red flags are that {reasons[0]}, and {reasons[1]}."
        else:
            joined = '; '.join(reasons[:-1]) + f'; and {reasons[-1]}'
            body   = f"The main red flags are: {joined}."

        return f"{opener} {body}"

    # ── Benign branch ─────────────────────────────────────────────────────────
    else:
        pct = int(round((1 - confidence) * 100))
        if risk_level == 'SAFE':
            opener = f"This email appears to be legitimate ({pct}% confidence)."
        else:
            opener = (
                f"This email is likely legitimate, though there is some uncertainty "
                f"({pct}% confidence)."
            )

        clean = []
        if not urgent_phrases:
            clean.append("no urgency tactics")
        if not credential_phrases:
            clean.append("no requests for passwords or sensitive data")
        if not suspicious_urls and fval('num_shortener_urls') == 0:
            clean.append("no suspicious links")
        if fval('sender_domain_mismatch') == 0 and fval('has_reply_to_mismatch') == 0:
            clean.append("no signs of sender impersonation")
        if fval('spf_dkim_fail') == 0:
            clean.append("it passes standard email authentication checks")

        if clean:
            if len(clean) == 1:
                body = f"Our analysis found {clean[0]}."
            else:
                body = "Our analysis found " + ', '.join(clean[:-1]) + f", and {clean[-1]}."
        else:
            body = "Our analysis did not detect any significant phishing patterns."

        caveats = []
        if urgent_phrases:
            sample = ', '.join(f'"{p}"' for p in urgent_phrases[:1])
            caveats.append(
                f"it does contain some urgent-sounding language ({sample}), "
                "which can also appear in legitimate emails"
            )
        if fval('num_shortener_urls') > 0:
            caveats.append(
                "some shortened links are present — always hover over links to verify "
                "the destination before clicking"
            )
        if fval('greeting_generic') > 0:
            caveats.append(
                'it uses a generic greeting, which is common in newsletters and automated emails'
            )

        if caveats:
            caveat_str = '; '.join(caveats) + '.'
            return f"{opener} {body} One thing to keep in mind: {caveat_str}"

        return f"{opener} {body}"


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
