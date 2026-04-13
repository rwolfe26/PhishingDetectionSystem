"""
URL Extractor Module

Extracts and analyzes URLs from email body text and HTML content.
Provides URL metadata useful for phishing detection.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse, parse_qs
import html.parser


@dataclass
class URLInfo:
    """Detailed information about an extracted URL."""
    raw_url: str
    scheme: str = ""
    domain: str = ""
    subdomain: str = ""
    tld: str = ""
    path: str = ""
    query: str = ""
    fragment: str = ""
    port: Optional[int] = None
    
    # Analysis fields
    anchor_text: str = ""  # Link text (if from HTML)
    is_ip_address: bool = False
    num_dots: int = 0
    num_dashes: int = 0
    path_depth: int = 0
    url_length: int = 0
    has_at_symbol: bool = False
    uses_https: bool = False
    has_suspicious_port: bool = False
    query_param_count: int = 0


class HTMLLinkParser(html.parser.HTMLParser):
    """Extract links and their anchor text from HTML."""
    
    def __init__(self):
        super().__init__()
        self.links: List[Tuple[str, str]] = []  # (url, anchor_text)
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []
    
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href' and value:
                    self._current_href = value
                    self._current_text = []
    
    def handle_endtag(self, tag: str):
        if tag == 'a' and self._current_href:
            anchor = ' '.join(self._current_text).strip()
            self.links.append((self._current_href, anchor))
            self._current_href = None
            self._current_text = []
    
    def handle_data(self, data: str):
        if self._current_href is not None:
            self._current_text.append(data)
    
    def error(self, message: str):
        pass  # Ignore HTML parsing errors


class URLExtractor:
    """
    Extracts URLs from email text and HTML content.
    
    Features:
    - Plain text URL extraction (http/https/ftp)
    - HTML href extraction with anchor text
    - URL parsing and analysis
    - Deobfuscation of common tricks
    """
    
    # URL patterns
    URL_PATTERN = re.compile(
        r'(?:https?://|ftp://|www\.)'  # Protocol or www
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?\.)*'  # Subdomains
        r'[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?'  # Domain
        r'(?:\.[a-zA-Z]{2,})'  # TLD
        r'(?::\d{1,5})?'  # Port
        r'(?:/[^\s<>\'"]*)?',  # Path
        re.IGNORECASE
    )
    
    # IP-based URL pattern
    IP_URL_PATTERN = re.compile(
        r'(?:https?://|ftp://)'
        r'(?:\d{1,3}\.){3}\d{1,3}'  # IPv4
        r'(?::\d{1,5})?'  # Port
        r'(?:/[^\s<>\'"]*)?',
        re.IGNORECASE
    )
    
    # Common TLDs for validation
    COMMON_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'co', 'io', 'ai', 'app', 'dev', 'xyz', 'info', 'biz',
        'uk', 'de', 'fr', 'jp', 'cn', 'ru', 'br', 'au', 'in',
        'ca', 'us', 'eu', 'it', 'es', 'nl', 'pl', 'be', 'ch'
    }
    
    # Suspicious ports (non-standard for web)
    SUSPICIOUS_PORTS = {81, 82, 83, 88, 8000, 8008, 8080, 8081, 8443, 8888, 9000}
    
    def __init__(self):
        self._html_parser = HTMLLinkParser()
    
    def extract_from_text(self, text: str) -> List[URLInfo]:
        """
        Extract URLs from plain text.
        
        Args:
            text: Plain text content
            
        Returns:
            List of URLInfo objects
        """
        urls = []
        seen = set()
        
        # Find standard URLs
        for match in self.URL_PATTERN.finditer(text):
            url = match.group(0)
            if url not in seen:
                seen.add(url)
                urls.append(self._analyze_url(url))
        
        # Find IP-based URLs
        for match in self.IP_URL_PATTERN.finditer(text):
            url = match.group(0)
            if url not in seen:
                seen.add(url)
                url_info = self._analyze_url(url)
                url_info.is_ip_address = True
                urls.append(url_info)
        
        return urls
    
    def extract_from_html(self, html_content: str) -> List[URLInfo]:
        """
        Extract URLs from HTML content, including anchor text.
        
        Args:
            html_content: HTML content
            
        Returns:
            List of URLInfo objects with anchor text
        """
        urls = []
        seen = set()
        
        # Parse HTML links
        parser = HTMLLinkParser()
        try:
            parser.feed(html_content)
        except Exception:
            pass  # Continue even if HTML is malformed
        
        for href, anchor in parser.links:
            if href and href not in seen and not href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                seen.add(href)
                url_info = self._analyze_url(href, anchor_text=anchor)
                urls.append(url_info)
        
        # Also extract URLs from text content (not in href)
        # Strip HTML tags first
        text_content = self._strip_html_tags(html_content)
        text_urls = self.extract_from_text(text_content)
        
        for url_info in text_urls:
            if url_info.raw_url not in seen:
                seen.add(url_info.raw_url)
                urls.append(url_info)
        
        return urls
    
    def extract_all(self, text: str = "", html: str = "") -> List[URLInfo]:
        """
        Extract URLs from both text and HTML content.
        
        Args:
            text: Plain text content
            html: HTML content
            
        Returns:
            Deduplicated list of URLInfo objects
        """
        urls = []
        seen = set()
        
        if html:
            for url_info in self.extract_from_html(html):
                if url_info.raw_url not in seen:
                    seen.add(url_info.raw_url)
                    urls.append(url_info)
        
        if text:
            for url_info in self.extract_from_text(text):
                if url_info.raw_url not in seen:
                    seen.add(url_info.raw_url)
                    urls.append(url_info)
        
        return urls
    
    def _analyze_url(self, url: str, anchor_text: str = "") -> URLInfo:
        """Analyze URL and extract features."""
        
        # Normalize URL
        normalized = url
        if not url.startswith(('http://', 'https://', 'ftp://')):
            normalized = 'http://' + url
        
        url_info = URLInfo(
            raw_url=url,
            anchor_text=anchor_text,
            url_length=len(url)
        )
        
        try:
            parsed = urlparse(normalized)
            
            url_info.scheme = parsed.scheme
            url_info.path = parsed.path
            url_info.query = parsed.query
            url_info.fragment = parsed.fragment
            url_info.uses_https = parsed.scheme == 'https'
            
            # Parse port
            if parsed.port:
                url_info.port = parsed.port
                url_info.has_suspicious_port = parsed.port in self.SUSPICIOUS_PORTS
            
            # Parse domain
            hostname = parsed.hostname or ""
            url_info.domain = hostname
            url_info.num_dots = hostname.count('.')
            url_info.num_dashes = hostname.count('-')
            
            # Check for IP address
            url_info.is_ip_address = self._is_ip_address(hostname)
            
            # Extract TLD and subdomain
            if not url_info.is_ip_address and '.' in hostname:
                parts = hostname.split('.')
                url_info.tld = parts[-1]
                if len(parts) > 2:
                    url_info.subdomain = '.'.join(parts[:-2])
            
            # Path analysis
            url_info.path_depth = len([p for p in parsed.path.split('/') if p])
            
            # Query parameters
            if parsed.query:
                url_info.query_param_count = len(parse_qs(parsed.query))
            
            # Check for @ symbol (potential credential stuffing)
            url_info.has_at_symbol = '@' in url
            
        except Exception:
            pass  # Keep defaults for malformed URLs
        
        return url_info
    
    def _is_ip_address(self, hostname: str) -> bool:
        """Check if hostname is an IP address."""
        parts = hostname.split('.')
        if len(parts) == 4:
            try:
                return all(0 <= int(p) <= 255 for p in parts)
            except ValueError:
                pass
        return False
    
    def _strip_html_tags(self, html: str) -> str:
        """Remove HTML tags from content."""
        # Simple regex-based tag removal
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<[^>]+>', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()


# Convenience functions
def extract_urls(text: str = "", html: str = "") -> List[URLInfo]:
    """Extract URLs from text and/or HTML content."""
    extractor = URLExtractor()
    return extractor.extract_all(text, html)


def get_unique_domains(urls: List[URLInfo]) -> Set[str]:
    """Get set of unique domains from URL list."""
    return {url.domain for url in urls if url.domain}
