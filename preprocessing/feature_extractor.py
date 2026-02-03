"""
Feature Extractor Module

Computes numeric features from parsed emails for Layer 2 model input.
Features aligned with existing datasets (Phishing_Legitimate_full.csv, email_phishing_data.csv).
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import Counter

from .email_parser import ParsedEmail
from .url_extractor import URLInfo, URLExtractor, get_unique_domains


@dataclass
class EmailFeatures:
    """Numeric features extracted from an email for ML model input."""
    
    # URL-based features (aligned with Phishing_Legitimate_full.csv)
    num_urls: int = 0
    num_unique_domains: int = 0
    has_ip_url: int = 0  # Binary: any URL uses IP address
    no_https_ratio: float = 0.0  # Ratio of non-HTTPS URLs
    avg_url_length: float = 0.0
    max_url_length: int = 0
    avg_path_depth: float = 0.0
    total_dots_in_urls: int = 0
    has_at_symbol_url: int = 0  # Binary: @ in any URL
    has_suspicious_port: int = 0  # Binary: non-standard port
    
    # Text-based features (aligned with email_phishing_data.csv)
    num_words: int = 0
    num_unique_words: int = 0
    num_chars: int = 0
    num_special_chars: int = 0
    
    # Urgency/phishing indicator features
    num_urgent_keywords: int = 0
    num_credential_keywords: int = 0
    num_action_keywords: int = 0
    
    # Header-based features
    has_reply_to_mismatch: int = 0  # Binary: Reply-To domain != From domain
    has_return_path_mismatch: int = 0  # Binary: Return-Path domain != From domain
    num_received_hops: int = 0
    has_suspicious_mailer: int = 0  # Binary: known phishing tools
    subject_has_urgent: int = 0  # Binary: urgency in subject
    subject_has_re_fw: int = 0  # Binary: Re:/Fw: in subject
    
    # HTML-based features
    has_html: int = 0  # Binary
    has_form: int = 0  # Binary: contains form tags
    has_iframe: int = 0  # Binary: contains iframe
    has_hidden_text: int = 0  # Binary: display:none or visibility:hidden
    num_external_links: int = 0
    link_text_url_mismatch: int = 0  # Binary: anchor text is URL different from href
    
    # Attachment features
    num_attachments: int = 0
    has_executable_attachment: int = 0  # Binary: .exe, .bat, .js, etc.
    has_archive_attachment: int = 0  # Binary: .zip, .rar, etc.
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for model input."""
        return {
            # URL features
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
            
            # Text features
            'num_words': self.num_words,
            'num_unique_words': self.num_unique_words,
            'num_chars': self.num_chars,
            'num_special_chars': self.num_special_chars,
            
            # Keyword features
            'num_urgent_keywords': self.num_urgent_keywords,
            'num_credential_keywords': self.num_credential_keywords,
            'num_action_keywords': self.num_action_keywords,
            
            # Header features
            'has_reply_to_mismatch': self.has_reply_to_mismatch,
            'has_return_path_mismatch': self.has_return_path_mismatch,
            'num_received_hops': self.num_received_hops,
            'has_suspicious_mailer': self.has_suspicious_mailer,
            'subject_has_urgent': self.subject_has_urgent,
            'subject_has_re_fw': self.subject_has_re_fw,
            
            # HTML features
            'has_html': self.has_html,
            'has_form': self.has_form,
            'has_iframe': self.has_iframe,
            'has_hidden_text': self.has_hidden_text,
            'num_external_links': self.num_external_links,
            'link_text_url_mismatch': self.link_text_url_mismatch,
            
            # Attachment features
            'num_attachments': self.num_attachments,
            'has_executable_attachment': self.has_executable_attachment,
            'has_archive_attachment': self.has_archive_attachment,
        }
    
    def to_list(self) -> List[float]:
        """Convert to list for model input (ordered)."""
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
    - Header-based: domain mismatches, hops, mailer
    - HTML-based: forms, iframes, hidden content
    - Attachment-based: count, dangerous types
    """
    
    # Keyword lists for detection
    URGENT_KEYWORDS = {
        'urgent', 'immediately', 'action required', 'act now', 'limited time',
        'expires', 'deadline', 'asap', 'important', 'alert', 'warning',
        'suspended', 'locked', 'unauthorized', 'verify now', 'confirm now',
        'final notice', 'last chance', 'dont delay', "don't delay"
    }
    
    CREDENTIAL_KEYWORDS = {
        'password', 'username', 'login', 'signin', 'sign in', 'sign-in',
        'credential', 'account', 'ssn', 'social security', 'bank account',
        'credit card', 'debit card', 'pin', 'security code', 'cvv',
        'routing number', 'verify your', 'confirm your', 'update your'
    }
    
    ACTION_KEYWORDS = {
        'click here', 'click below', 'click the link', 'download',
        'open attachment', 'see attached', 'login now', 'sign in now',
        'verify account', 'confirm identity', 'reset password', 'unlock',
        'restore access', 'reactivate', 'update payment', 'enter details'
    }
    
    SUSPICIOUS_MAILERS = {
        'king phisher', 'gophish', 'setoolkit', 'beef', 'phishing frenzy',
        'lucy', 'hidden cobra', 'mailchimp-phish'  # Add known malicious tools
    }
    
    EXECUTABLE_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.msi', '.js', '.jse', '.vbs', '.vbe',
        '.wsf', '.wsh', '.ps1', '.psm1', '.scr', '.hta', '.jar', '.pif'
    }
    
    ARCHIVE_EXTENSIONS = {
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.iso', '.cab'
    }
    
    def __init__(self):
        self._url_extractor = URLExtractor()
    
    def extract(self, parsed_email: ParsedEmail, urls: Optional[List[URLInfo]] = None) -> EmailFeatures:
        """
        Extract all features from a parsed email.
        
        Args:
            parsed_email: ParsedEmail object
            urls: Pre-extracted URLs (optional, will extract if not provided)
            
        Returns:
            EmailFeatures object with computed features
        """
        features = EmailFeatures()
        
        # Extract URLs if not provided
        if urls is None:
            urls = self._url_extractor.extract_all(
                text=parsed_email.body_text,
                html=parsed_email.body_html
            )
        
        # Compute feature groups
        self._extract_url_features(urls, features)
        self._extract_text_features(parsed_email, features)
        self._extract_keyword_features(parsed_email, features)
        self._extract_header_features(parsed_email, features)
        self._extract_html_features(parsed_email, urls, features)
        self._extract_attachment_features(parsed_email, features)
        
        return features
    
    def _extract_url_features(self, urls: List[URLInfo], features: EmailFeatures):
        """Extract URL-based features."""
        features.num_urls = len(urls)
        features.num_unique_domains = len(get_unique_domains(urls))
        
        if not urls:
            return
        
        # Binary indicators
        features.has_ip_url = int(any(u.is_ip_address for u in urls))
        features.has_at_symbol_url = int(any(u.has_at_symbol for u in urls))
        features.has_suspicious_port = int(any(u.has_suspicious_port for u in urls))
        
        # HTTPS ratio
        non_https = sum(1 for u in urls if not u.uses_https)
        features.no_https_ratio = non_https / len(urls)
        
        # Length statistics
        lengths = [u.url_length for u in urls]
        features.avg_url_length = sum(lengths) / len(lengths)
        features.max_url_length = max(lengths)
        
        # Path depth
        depths = [u.path_depth for u in urls]
        features.avg_path_depth = sum(depths) / len(depths)
        
        # Dots in domains
        features.total_dots_in_urls = sum(u.num_dots for u in urls)
    
    def _extract_text_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        """Extract text-based features."""
        # Use body text, fall back to HTML stripped
        text = parsed_email.body_text or self._strip_html(parsed_email.body_html)
        
        if not text:
            return
        
        # Word analysis
        words = re.findall(r'\b\w+\b', text.lower())
        features.num_words = len(words)
        features.num_unique_words = len(set(words))
        
        # Character analysis
        features.num_chars = len(text)
        features.num_special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    
    def _extract_keyword_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        """Extract keyword-based features."""
        # Combine text sources
        text = ' '.join([
            parsed_email.subject,
            parsed_email.body_text,
            self._strip_html(parsed_email.body_html)
        ]).lower()
        
        # Count keyword matches
        features.num_urgent_keywords = sum(1 for kw in self.URGENT_KEYWORDS if kw in text)
        features.num_credential_keywords = sum(1 for kw in self.CREDENTIAL_KEYWORDS if kw in text)
        features.num_action_keywords = sum(1 for kw in self.ACTION_KEYWORDS if kw in text)
    
    def _extract_header_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        """Extract header-based features."""
        # Domain extraction helper
        def get_domain(address: str) -> str:
            match = re.search(r'@([\w.-]+)', address)
            return match.group(1).lower() if match else ""
        
        from_domain = get_domain(parsed_email.from_address)
        
        # Reply-To mismatch
        if parsed_email.reply_to:
            reply_domain = get_domain(parsed_email.reply_to)
            features.has_reply_to_mismatch = int(reply_domain and reply_domain != from_domain)
        
        # Return-Path mismatch
        if parsed_email.return_path:
            return_domain = get_domain(parsed_email.return_path)
            features.has_return_path_mismatch = int(return_domain and return_domain != from_domain)
        
        # Received hops
        features.num_received_hops = len(parsed_email.received_chain)
        
        # Suspicious mailer
        mailer = (parsed_email.x_mailer or "").lower()
        features.has_suspicious_mailer = int(any(s in mailer for s in self.SUSPICIOUS_MAILERS))
        
        # Subject analysis
        subject_lower = parsed_email.subject.lower()
        features.subject_has_urgent = int(any(kw in subject_lower for kw in self.URGENT_KEYWORDS))
        features.subject_has_re_fw = int(bool(re.match(r'^(re:|fw:|fwd:)', subject_lower)))
    
    def _extract_html_features(self, parsed_email: ParsedEmail, urls: List[URLInfo], features: EmailFeatures):
        """Extract HTML-based features."""
        html = parsed_email.body_html
        
        if not html:
            return
        
        features.has_html = 1
        html_lower = html.lower()
        
        # Form detection
        features.has_form = int('<form' in html_lower)
        
        # Iframe detection
        features.has_iframe = int('<iframe' in html_lower)
        
        # Hidden text detection
        hidden_patterns = [
            r'display\s*:\s*none',
            r'visibility\s*:\s*hidden',
            r'font-size\s*:\s*0',
            r'height\s*:\s*0',
            r'width\s*:\s*0'
        ]
        features.has_hidden_text = int(any(re.search(p, html_lower) for p in hidden_patterns))
        
        # External links (simplified: any non-relative URL)
        features.num_external_links = len([u for u in urls if u.scheme in ('http', 'https')])
        
        # Link text vs URL mismatch
        for url_info in urls:
            anchor = url_info.anchor_text.lower().strip()
            if anchor and re.match(r'^https?://', anchor):
                # Anchor text looks like a URL
                if anchor not in url_info.raw_url.lower():
                    features.link_text_url_mismatch = 1
                    break
    
    def _extract_attachment_features(self, parsed_email: ParsedEmail, features: EmailFeatures):
        """Extract attachment-based features."""
        attachments = parsed_email.attachments
        features.num_attachments = len(attachments)
        
        for att in attachments:
            filename = (att.filename or "").lower()
            
            # Executable check
            if any(filename.endswith(ext) for ext in self.EXECUTABLE_EXTENSIONS):
                features.has_executable_attachment = 1
            
            # Archive check
            if any(filename.endswith(ext) for ext in self.ARCHIVE_EXTENSIONS):
                features.has_archive_attachment = 1
    
    def _strip_html(self, html: str) -> str:
        """Remove HTML tags from content."""
        if not html:
            return ""
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<[^>]+>', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()


# Convenience function
def extract_features(parsed_email: ParsedEmail, urls: Optional[List[URLInfo]] = None) -> EmailFeatures:
    """Extract numeric features from a parsed email."""
    extractor = FeatureExtractor()
    return extractor.extract(parsed_email, urls)
