"""
Feature Extractor Module

Computes numeric features from parsed emails for phishing detection.
Features cover URL analysis, text content, headers, HTML structure, attachments,
and dedicated phishing signals (brand impersonation, SPF/DKIM, urgency density, etc.).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .email_parser import ParsedEmail
from .url_extractor import URLInfo, URLExtractor, get_unique_domains


@dataclass
class EmailFeatures:
    """Numeric features extracted from an email for ML model input."""

    # ── URL-based features ──────────────────────────────────────────────────
    num_urls: int = 0
    num_unique_domains: int = 0
    has_ip_url: int = 0
    no_https_ratio: float = 0.0
    avg_url_length: float = 0.0
    max_url_length: int = 0
    avg_path_depth: float = 0.0
    total_dots_in_urls: int = 0
    has_at_symbol_url: int = 0
    has_suspicious_port: int = 0

    # ── Text-based features ─────────────────────────────────────────────────
    num_words: int = 0
    num_unique_words: int = 0
    num_chars: int = 0
    num_special_chars: int = 0
    special_char_ratio: float = 0.0   # special chars / total chars (length-independent)
    unique_word_ratio: float = 0.0    # unique words / total words (lexical diversity)
    caps_ratio: float = 0.0           # uppercase letters / total letters

    # ── Urgency / phishing keyword features ────────────────────────────────
    num_urgent_keywords: int = 0
    num_credential_keywords: int = 0
    num_action_keywords: int = 0

    # ── Header-based features ───────────────────────────────────────────────
    has_reply_to_mismatch: int = 0
    has_return_path_mismatch: int = 0
    num_received_hops: int = 0
    has_suspicious_mailer: int = 0
    subject_has_urgent: int = 0
    subject_has_re_fw: int = 0

    # ── HTML-based features ─────────────────────────────────────────────────
    has_html: int = 0
    has_form: int = 0
    has_iframe: int = 0
    has_hidden_text: int = 0
    num_external_links: int = 0
    link_text_url_mismatch: int = 0

    # ── Attachment features ─────────────────────────────────────────────────
    num_attachments: int = 0
    has_executable_attachment: int = 0
    has_archive_attachment: int = 0

    # ── Phishing-specific signals (new) ────────────────────────────────────
    spf_dkim_fail: int = 0           # Authentication-Results shows fail/none
    sender_domain_mismatch: int = 0  # Display name domain ≠ actual From domain
    num_homograph_chars: int = 0     # IDN/look-alike chars in URL domains
    brand_impersonation_score: float = 0.0  # Min Levenshtein to known brand domains
    urgency_density: float = 0.0     # Urgency keyword count / total words
    html_text_ratio: float = 0.0     # len(html) / (len(text)+1)
    num_shortener_urls: int = 0      # Count of URL-shortener domains
    greeting_generic: int = 0        # "Dear Customer/User/Sir" detected
    num_auth_keywords: int = 0       # "verify", "confirm", "authenticate", etc.
    subject_all_caps_ratio: float = 0.0  # Fraction of subject words in ALL CAPS

    # ── URL redirect resolution signals ────────────────────────────────────
    has_redirect_url: int = 0        # Any shortener resolved to a different domain
    num_redirect_hops: int = 0       # Total redirect hops across all resolved URLs

    def to_dict(self) -> Dict[str, float]:
        """Return ordered dict for model input."""
        return {
            # URL
            'num_urls': self.num_urls,
            'num_unique_domains': self.num_unique_domains,
            'has_ip_url': self.has_ip_url,
            'no_https_ratio': self.no_https_ratio,
            'avg_url_length': self.avg_url_length,
            'max_url_length': self.max_url_length,
            'avg_path_depth': self.avg_path_depth,
            'total_dots_in_urls': self.total_dots_in_urls,
            'has_at_symbol_url': self.has_at_symbol_url,
            'has_suspicious_port': self.has_suspicious_port,
            # Text
            'num_words': self.num_words,
            'num_unique_words': self.num_unique_words,
            'num_chars': self.num_chars,
            'num_special_chars': self.num_special_chars,
            'special_char_ratio': self.special_char_ratio,
            'unique_word_ratio': self.unique_word_ratio,
            'caps_ratio': self.caps_ratio,
            # Keywords
            'num_urgent_keywords': self.num_urgent_keywords,
            'num_credential_keywords': self.num_credential_keywords,
            'num_action_keywords': self.num_action_keywords,
            # Headers
            'has_reply_to_mismatch': self.has_reply_to_mismatch,
            'has_return_path_mismatch': self.has_return_path_mismatch,
            'num_received_hops': self.num_received_hops,
            'has_suspicious_mailer': self.has_suspicious_mailer,
            'subject_has_urgent': self.subject_has_urgent,
            'subject_has_re_fw': self.subject_has_re_fw,
            # HTML
            'has_html': self.has_html,
            'has_form': self.has_form,
            'has_iframe': self.has_iframe,
            'has_hidden_text': self.has_hidden_text,
            'num_external_links': self.num_external_links,
            'link_text_url_mismatch': self.link_text_url_mismatch,
            # Attachments
            'num_attachments': self.num_attachments,
            'has_executable_attachment': self.has_executable_attachment,
            'has_archive_attachment': self.has_archive_attachment,
            # Phishing-specific
            'spf_dkim_fail': self.spf_dkim_fail,
            'sender_domain_mismatch': self.sender_domain_mismatch,
            'num_homograph_chars': self.num_homograph_chars,
            'brand_impersonation_score': self.brand_impersonation_score,
            'urgency_density': self.urgency_density,
            'html_text_ratio': self.html_text_ratio,
            'num_shortener_urls': self.num_shortener_urls,
            'greeting_generic': self.greeting_generic,
            'num_auth_keywords': self.num_auth_keywords,
            'subject_all_caps_ratio': self.subject_all_caps_ratio,
            # URL redirect resolution
            'has_redirect_url': self.has_redirect_url,
            'num_redirect_hops': self.num_redirect_hops,
        }

    def to_list(self) -> List[float]:
        """Convert to ordered list for model input."""
        return list(self.to_dict().values())

    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        return list(EmailFeatures().to_dict().keys())


class FeatureExtractor:
    """
    Extracts numeric features from parsed emails.

    Feature categories:
    - URL-based: count, protocols, domains, structure
    - Text-based: word counts, character analysis
    - Keyword-based: urgency, credentials, action words
    - Header-based: domain mismatches, hops, mailer, SPF/DKIM
    - HTML-based: forms, iframes, hidden content
    - Attachment-based: count, dangerous types
    - Phishing-specific: brand impersonation, homographs, shorteners, etc.
    """

    # ── Keyword sets ────────────────────────────────────────────────────────

    URGENT_KEYWORDS = {
        'urgent', 'immediately', 'action required', 'act now', 'limited time',
        'expires', 'deadline', 'asap', 'important', 'alert', 'warning',
        'suspended', 'locked', 'unauthorized', 'verify now', 'confirm now',
        'final notice', 'last chance', "don't delay", 'dont delay',
        'your account will', 'will be closed', 'will be suspended',
        'within 24 hours', 'within 48 hours',
    }

    CREDENTIAL_KEYWORDS = {
        'password', 'username', 'login', 'signin', 'sign in', 'sign-in',
        'credential', 'ssn', 'social security', 'bank account',
        'credit card', 'debit card', 'pin', 'security code', 'cvv',
        'routing number', 'verify your', 'confirm your', 'update your',
    }

    ACTION_KEYWORDS = {
        'click here', 'click below', 'click the link', 'download',
        'open attachment', 'see attached', 'login now', 'sign in now',
        'verify account', 'confirm identity', 'reset password', 'unlock',
        'restore access', 'reactivate', 'update payment', 'enter details',
    }

    AUTH_KEYWORDS = {
        'verify', 'authenticate', 'confirm', 'validate', 'authorize',
        'two-factor', '2fa', 'one-time', 'otp', 'security question',
        'identity verification', 'account verification',
    }

    GENERIC_GREETINGS = {
        'dear customer', 'dear user', 'dear client', 'dear member',
        'dear account holder', 'dear valued customer', 'dear sir',
        'dear madam', 'dear sir or madam', 'hello customer',
        'dear friend', 'greetings',
    }

    SUSPICIOUS_MAILERS = {
        'king phisher', 'gophish', 'setoolkit', 'beef', 'phishing frenzy',
        'lucy', 'hidden cobra', 'mailchimp-phish',
    }

    EXECUTABLE_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.msi', '.js', '.jse', '.vbs', '.vbe',
        '.wsf', '.wsh', '.ps1', '.psm1', '.scr', '.hta', '.jar', '.pif',
    }

    ARCHIVE_EXTENSIONS = {
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.iso', '.cab',
    }

    # Extensions that appear as the "decoy" first extension in double-extension attacks
    # e.g. invoice.pdf.exe or report.docx.js
    _DOUBLE_EXT_DECOYS = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.txt', '.jpg', '.png', '.mp3', '.mp4',
    }

    # Known brand domains for impersonation detection
    BRAND_DOMAINS = [
        'paypal.com', 'amazon.com', 'google.com', 'microsoft.com', 'apple.com',
        'facebook.com', 'instagram.com', 'twitter.com', 'linkedin.com',
        'netflix.com', 'ebay.com', 'bankofamerica.com', 'chase.com',
        'wellsfargo.com', 'citibank.com', 'irs.gov', 'fedex.com', 'ups.com',
        'usps.com', 'dhl.com', 'dropbox.com', 'onedrive.com', 'icloud.com',
    ]

    # URL shortener domains
    URL_SHORTENERS = {
        'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd',
        'buff.ly', 'adf.ly', 'tiny.cc', 'rebrand.ly', 'shorturl.at',
        'cutt.ly', 'rb.gy', 'tr.im', 'shrinkme.io',
    }

    # Confusable Unicode character categories (used in IDN homograph attacks)
    _CONFUSABLE_SCRIPT_CATEGORIES = {'Ll', 'Lu', 'Nd'}

    def __init__(self):
        self._url_extractor = URLExtractor()

    def extract(self, parsed_email: ParsedEmail,
                urls: Optional[List[URLInfo]] = None) -> EmailFeatures:
        """
        Extract all features from a parsed email.

        Args:
            parsed_email: ParsedEmail object
            urls: Pre-extracted URLs (optional)

        Returns:
            EmailFeatures object
        """
        features = EmailFeatures()

        if urls is None:
            urls = self._url_extractor.extract_all(
                text=parsed_email.body_text,
                html=parsed_email.body_html
            )

        self._extract_url_features(urls, features)
        self._extract_text_features(parsed_email, features)
        self._extract_keyword_features(parsed_email, features)
        self._extract_header_features(parsed_email, features)
        self._extract_html_features(parsed_email, urls, features)
        self._extract_attachment_features(parsed_email, features)
        self._extract_phishing_signals(parsed_email, urls, features)

        return features

    # ── Feature group extractors ────────────────────────────────────────────

    def _extract_url_features(self, urls: List[URLInfo], features: EmailFeatures):
        features.num_urls = len(urls)
        features.num_unique_domains = len(get_unique_domains(urls))

        if not urls:
            return

        features.has_ip_url = int(any(u.is_ip_address for u in urls))
        features.has_at_symbol_url = int(any(u.has_at_symbol for u in urls))
        features.has_suspicious_port = int(any(u.has_suspicious_port for u in urls))

        non_https = sum(1 for u in urls if not u.uses_https)
        features.no_https_ratio = non_https / len(urls)

        lengths = [u.url_length for u in urls]
        features.avg_url_length = sum(lengths) / len(lengths)
        features.max_url_length = max(lengths)

        depths = [u.path_depth for u in urls]
        features.avg_path_depth = sum(depths) / len(depths)

        features.total_dots_in_urls = sum(u.num_dots for u in urls)

    def _extract_text_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        text = parsed_email.body_text or self._strip_html(parsed_email.body_html)
        if not text:
            return

        words = re.findall(r'\b\w+\b', text.lower())
        features.num_words = len(words)
        features.num_unique_words = len(set(words))
        features.num_chars = len(text)
        features.num_special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))

        # Length-independent ratios — these give the model scale-free signals
        if features.num_chars > 0:
            features.special_char_ratio = round(features.num_special_chars / features.num_chars, 6)
            letters = re.findall(r'[a-zA-Z]', text)
            upper = re.findall(r'[A-Z]', text)
            features.caps_ratio = round(len(upper) / len(letters), 6) if letters else 0.0
        if features.num_words > 0:
            features.unique_word_ratio = round(features.num_unique_words / features.num_words, 6)

    def _extract_keyword_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        text = ' '.join([
            parsed_email.subject,
            parsed_email.body_text,
            self._strip_html(parsed_email.body_html)
        ]).lower()

        features.num_urgent_keywords = sum(1 for kw in self.URGENT_KEYWORDS if kw in text)
        features.num_credential_keywords = sum(
            1 for kw in self.CREDENTIAL_KEYWORDS if kw in text
        )
        features.num_action_keywords = sum(1 for kw in self.ACTION_KEYWORDS if kw in text)

    def _extract_header_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        def get_domain(address: str) -> str:
            m = re.search(r'@([\w.-]+)', address)
            return m.group(1).lower() if m else ''

        from_domain = get_domain(parsed_email.from_address)

        if parsed_email.reply_to:
            reply_domain = get_domain(parsed_email.reply_to)
            features.has_reply_to_mismatch = int(
                bool(reply_domain and reply_domain != from_domain)
            )

        if parsed_email.return_path:
            return_domain = get_domain(parsed_email.return_path)
            features.has_return_path_mismatch = int(
                bool(return_domain and return_domain != from_domain)
            )

        features.num_received_hops = len(parsed_email.received_chain)

        mailer = (parsed_email.x_mailer or '').lower()
        features.has_suspicious_mailer = int(any(s in mailer for s in self.SUSPICIOUS_MAILERS))

        subject_lower = parsed_email.subject.lower()
        features.subject_has_urgent = int(
            any(kw in subject_lower for kw in self.URGENT_KEYWORDS)
        )
        features.subject_has_re_fw = int(bool(re.match(r'^(re:|fw:|fwd:)', subject_lower)))

    def _extract_html_features(self, parsed_email: ParsedEmail,
                               urls: List[URLInfo], features: EmailFeatures):
        html = parsed_email.body_html
        if not html:
            return

        features.has_html = 1
        html_lower = html.lower()

        features.has_form = int('<form' in html_lower)
        features.has_iframe = int('<iframe' in html_lower)

        hidden_patterns = [
            r'display\s*:\s*none',
            r'visibility\s*:\s*hidden',
            r'font-size\s*:\s*0',
            r'height\s*:\s*0',
            r'width\s*:\s*0',
        ]
        features.has_hidden_text = int(
            any(re.search(p, html_lower) for p in hidden_patterns)
        )

        features.num_external_links = len(
            [u for u in urls if u.scheme in ('http', 'https')]
        )

        for url_info in urls:
            anchor = url_info.anchor_text.lower().strip()
            if anchor and re.match(r'^https?://', anchor):
                if anchor not in url_info.raw_url.lower():
                    features.link_text_url_mismatch = 1
                    break

    def _extract_attachment_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        attachments = parsed_email.attachments
        features.num_attachments = len(attachments)

        for att in attachments:
            filename = (att.filename or '').lower().strip()
            if not filename:
                continue

            is_exec = any(filename.endswith(ext) for ext in self.EXECUTABLE_EXTENSIONS)

            # Double-extension attack: e.g. "invoice.pdf.exe" or "report.docx.js"
            # Detect by checking whether the second-to-last extension is a known decoy
            parts = filename.rsplit('.', 2)
            if not is_exec and len(parts) == 3:
                decoy_ext = '.' + parts[1]
                final_ext = '.' + parts[2]
                if decoy_ext in self._DOUBLE_EXT_DECOYS and \
                        final_ext in self.EXECUTABLE_EXTENSIONS:
                    is_exec = True

            if is_exec:
                features.has_executable_attachment = 1
            if any(filename.endswith(ext) for ext in self.ARCHIVE_EXTENSIONS):
                features.has_archive_attachment = 1

    def _extract_phishing_signals(self, parsed_email: ParsedEmail,
                                   urls: List[URLInfo], features: EmailFeatures):
        """Extract advanced phishing-specific features."""

        # SPF / DKIM / DMARC failure
        auth_result = ''
        for key, value in parsed_email.headers.items():
            if key.lower() in ('authentication-results', 'arc-authentication-results'):
                val = value if isinstance(value, str) else ' '.join(value)
                auth_result += ' ' + val.lower()
        if auth_result:
            fail_patterns = ['spf=fail', 'spf=none', 'dkim=fail', 'dkim=none',
                             'dmarc=fail', 'dmarc=none']
            pass_patterns = ['spf=pass', 'dkim=pass', 'dmarc=pass']
            has_fail = any(p in auth_result for p in fail_patterns)
            has_pass = any(p in auth_result for p in pass_patterns)
            if has_fail and not has_pass:
                features.spf_dkim_fail = 1

        # Display name domain mismatch — "PayPal <attacker@evil.com>"
        from_raw = parsed_email.from_address
        display_match = re.match(r'^([^<]+)<([^>]+)>', from_raw)
        if display_match:
            display_name = display_match.group(1).strip().lower()
            actual_addr = display_match.group(2).strip()
            actual_domain_m = re.search(r'@([\w.-]+)', actual_addr)
            if actual_domain_m:
                actual_domain = actual_domain_m.group(1).lower()
                for brand in self.BRAND_DOMAINS:
                    brand_name = brand.split('.')[0]
                    if brand_name in display_name and brand not in actual_domain:
                        features.sender_domain_mismatch = 1
                        break

        # Homograph characters — non-ASCII in URL domains (cap at first 10 URLs)
        total_homograph = 0
        for url_info in urls[:10]:
            total_homograph += sum(1 for c in url_info.domain if ord(c) > 127)
        features.num_homograph_chars = total_homograph

        # Brand impersonation score — Levenshtein on first 5 URLs, length-filtered
        if urls:
            min_norm = 1.0
            for url_info in urls[:5]:
                domain = url_info.domain.lower()
                if not domain:
                    continue
                domain_root = domain.split('.')[0] if '.' in domain else domain
                for brand in self.BRAND_DOMAINS:
                    if domain == brand:
                        continue
                    brand_root = brand.split('.')[0]
                    # Only compare if length difference is small (fast pre-filter)
                    if abs(len(domain_root) - len(brand_root)) > 3:
                        continue
                    dist = self._levenshtein(domain_root, brand_root)
                    max_len = max(len(domain_root), len(brand_root), 1)
                    norm = dist / max_len
                    if 0 < norm < min_norm:
                        min_norm = norm
            features.brand_impersonation_score = round(1.0 - min_norm, 4)

        # Urgency density
        if features.num_words > 0:
            features.urgency_density = round(
                features.num_urgent_keywords / features.num_words, 6
            )

        # HTML-to-text ratio (capped to avoid extreme values)
        html_len = len(parsed_email.body_html or '')
        text_len = len(parsed_email.body_text or '')
        features.html_text_ratio = round(min(html_len / (text_len + 1), 100.0), 4)

        # URL shortener count
        features.num_shortener_urls = sum(
            1 for u in urls if u.domain.lower() in self.URL_SHORTENERS
        )

        # Generic greeting detection — search first 500 chars of body
        body_start = (parsed_email.body_text or '')[:500].lower()
        features.greeting_generic = int(
            any(g in body_start for g in self.GENERIC_GREETINGS)
        )

        # Authentication action keywords
        all_text = ' '.join([
            parsed_email.subject,
            parsed_email.body_text,
            self._strip_html(parsed_email.body_html),
        ]).lower()
        features.num_auth_keywords = sum(1 for kw in self.AUTH_KEYWORDS if kw in all_text)

        # Subject all-caps ratio
        subject_words = parsed_email.subject.split()
        if subject_words:
            caps_words = sum(1 for w in subject_words if w.isupper() and len(w) > 1)
            features.subject_all_caps_ratio = round(caps_words / len(subject_words), 4)

        # URL redirect resolution — populated when URLRedirectResolver has been called
        total_hops = 0
        any_domain_changed = False
        for url_info in urls:
            if url_info.num_redirects > 0:
                total_hops += url_info.num_redirects
                if url_info.resolved_domain and url_info.resolved_domain != url_info.domain:
                    any_domain_changed = True
        features.has_redirect_url = int(any_domain_changed)
        features.num_redirect_hops = total_hops

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            new_row = [i]
            for j, cb in enumerate(b, 1):
                new_row.append(min(
                    row[j] + 1,       # deletion
                    new_row[j - 1] + 1,  # insertion
                    row[j - 1] + (ca != cb),  # substitution
                ))
            row = new_row
        return row[-1]

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags from content."""
        if not html:
            return ''
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html,
                       flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean,
                       flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<[^>]+>', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()


def extract_features(parsed_email: ParsedEmail,
                     urls: Optional[List[URLInfo]] = None) -> EmailFeatures:
    """Extract numeric features from a parsed email."""
    return FeatureExtractor().extract(parsed_email, urls)
