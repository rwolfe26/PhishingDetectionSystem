"""
Email Parser Module

Parses raw email text (RFC 822 / mbox format) into structured components:
- Headers (dict)
- Subject
- Body (plain text)
- HTML body
- Attachments metadata
"""

import email
from email import policy
from email.parser import Parser, BytesParser
from email.message import EmailMessage
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re


@dataclass
class AttachmentInfo:
    """Metadata about an email attachment."""
    filename: Optional[str]
    content_type: str
    size: int
    content_disposition: Optional[str] = None


@dataclass
class ParsedEmail:
    """Structured representation of a parsed email."""
    # Core fields
    subject: str = ""
    body_text: str = ""
    body_html: str = ""
    
    # Headers
    headers: Dict[str, Any] = field(default_factory=dict)
    from_address: str = ""
    to_addresses: List[str] = field(default_factory=list)
    cc_addresses: List[str] = field(default_factory=list)
    reply_to: str = ""
    return_path: str = ""
    date: str = ""
    message_id: str = ""
    
    # Technical headers
    received_chain: List[str] = field(default_factory=list)
    x_mailer: str = ""
    content_type: str = ""
    
    # Attachments
    attachments: List[AttachmentInfo] = field(default_factory=list)
    
    # Parsing metadata
    is_multipart: bool = False
    parse_errors: List[str] = field(default_factory=list)


class EmailParser:
    """
    Parser for raw email text into structured components.
    
    Handles:
    - Plain text emails (RFC 822)
    - MIME multipart emails
    - Various encodings
    - Malformed headers
    """
    
    def __init__(self):
        self._parser = Parser(policy=policy.default)
    
    def parse(self, raw_email: str) -> ParsedEmail:
        """
        Parse raw email text into structured components.
        
        Args:
            raw_email: Raw email text (headers + body)
            
        Returns:
            ParsedEmail object with extracted components
        """
        result = ParsedEmail()
        
        try:
            # Handle mbox "From " line if present
            raw_email = self._strip_mbox_header(raw_email)
            
            # Parse email
            msg = self._parser.parsestr(raw_email)
            
            # Extract headers
            result.headers = self._extract_headers(msg)
            result.subject = self._decode_header(msg.get('Subject', ''))
            result.from_address = self._decode_header(msg.get('From', ''))
            result.to_addresses = self._parse_address_list(msg.get('To', ''))
            result.cc_addresses = self._parse_address_list(msg.get('Cc', ''))
            result.reply_to = self._decode_header(msg.get('Reply-To', ''))
            result.return_path = self._decode_header(msg.get('Return-Path', ''))
            result.date = msg.get('Date', '')
            result.message_id = msg.get('Message-ID', '')
            result.x_mailer = msg.get('X-Mailer', '') or msg.get('User-Agent', '')
            result.content_type = msg.get_content_type()
            result.is_multipart = msg.is_multipart()
            
            # Extract Received chain for hop analysis
            result.received_chain = msg.get_all('Received', [])
            
            # Extract body and attachments
            self._extract_body_and_attachments(msg, result)
            
        except Exception as e:
            result.parse_errors.append(f"Parse error: {str(e)}")
            # Fallback: treat entire content as body
            result.body_text = raw_email
            
        return result
    
    def parse_bytes(self, raw_bytes: bytes, encoding: str = 'utf-8') -> ParsedEmail:
        """
        Parse raw email bytes into structured components.
        
        Args:
            raw_bytes: Raw email bytes
            encoding: Fallback encoding if not specified in email
            
        Returns:
            ParsedEmail object with extracted components
        """
        try:
            raw_email = raw_bytes.decode(encoding, errors='replace')
        except Exception:
            raw_email = raw_bytes.decode('latin-1', errors='replace')
        
        return self.parse(raw_email)
    
    def _strip_mbox_header(self, raw_email: str) -> str:
        """Remove mbox 'From ' line if present."""
        lines = raw_email.split('\n', 1)
        if lines and lines[0].startswith('From '):
            return lines[1] if len(lines) > 1 else ''
        return raw_email
    
    def _extract_headers(self, msg: EmailMessage) -> Dict[str, Any]:
        """Extract all headers as a dictionary."""
        headers = {}
        for key in msg.keys():
            values = msg.get_all(key, [])
            if len(values) == 1:
                headers[key] = self._decode_header(values[0])
            else:
                headers[key] = [self._decode_header(v) for v in values]
        return headers
    
    def _decode_header(self, header_value: Any) -> str:
        """Decode header value, handling encoded words."""
        if header_value is None:
            return ""
        
        if isinstance(header_value, str):
            return header_value
        
        try:
            # Handle email.header.Header objects
            decoded_parts = email.header.decode_header(str(header_value))
            result = []
            for part, charset in decoded_parts:
                if isinstance(part, bytes):
                    charset = charset or 'utf-8'
                    result.append(part.decode(charset, errors='replace'))
                else:
                    result.append(part)
            return ' '.join(result)
        except Exception:
            return str(header_value)
    
    def _parse_address_list(self, address_str: str) -> List[str]:
        """Parse comma-separated address list."""
        if not address_str:
            return []
        
        decoded = self._decode_header(address_str)
        # Split on comma, but be careful of quoted strings
        addresses = []
        current = []
        in_quotes = False
        
        for char in decoded:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                addr = ''.join(current).strip()
                if addr:
                    addresses.append(addr)
                current = []
                continue
            current.append(char)
        
        # Don't forget the last address
        addr = ''.join(current).strip()
        if addr:
            addresses.append(addr)
        
        return addresses
    
    def _extract_body_and_attachments(self, msg: EmailMessage, result: ParsedEmail):
        """Extract body text, HTML, and attachment metadata."""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get_content_disposition()
                
                # Check for attachments
                if content_disposition == 'attachment':
                    self._add_attachment(part, result)
                elif content_type == 'text/plain' and not result.body_text:
                    result.body_text = self._get_payload_text(part)
                elif content_type == 'text/html' and not result.body_html:
                    result.body_html = self._get_payload_text(part)
                elif content_disposition == 'inline' and content_type.startswith('image/'):
                    # Inline images - treat as attachment metadata
                    self._add_attachment(part, result)
        else:
            # Single part message
            content_type = msg.get_content_type()
            payload = self._get_payload_text(msg)
            
            if content_type == 'text/html':
                result.body_html = payload
            else:
                result.body_text = payload
    
    def _get_payload_text(self, part: EmailMessage) -> str:
        """Extract text payload from message part."""
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return str(part.get_payload()) if part.get_payload() else ""
            
            # Try to decode with charset
            charset = part.get_content_charset() or 'utf-8'
            try:
                return payload.decode(charset, errors='replace')
            except (LookupError, TypeError):
                return payload.decode('utf-8', errors='replace')
        except Exception:
            return ""
    
    def _add_attachment(self, part: EmailMessage, result: ParsedEmail):
        """Add attachment metadata to result."""
        try:
            payload = part.get_payload(decode=True)
            size = len(payload) if payload else 0
        except Exception:
            size = 0
        
        attachment = AttachmentInfo(
            filename=part.get_filename(),
            content_type=part.get_content_type(),
            size=size,
            content_disposition=part.get_content_disposition()
        )
        result.attachments.append(attachment)


# Convenience function
def parse_email(raw_email: str) -> ParsedEmail:
    """Parse raw email text into structured components."""
    parser = EmailParser()
    return parser.parse(raw_email)
