/**
 * PhishingDetector — Content Script
 *
 * Runs on Gmail and Outlook Web. Watches for email opens, extracts
 * headers + body, calls the PhishingDetector API, and injects a
 * result badge at the top of the email.
 */

(function () {
  'use strict';

  const DEFAULT_API = 'https://rwolfe26-phishing-detector.hf.space';
  const BADGE_ID    = 'pd-badge';
  const SCAN_KEY    = 'pdScanned'; // dataset key — maps to data-pd-scanned

  // ── Identify which client we're in ────────────────────────────────────────
  const host   = location.hostname;
  const CLIENT =
    host === 'mail.google.com'                                              ? 'gmail'
    : ['outlook.live.com', 'outlook.office.com', 'outlook.office365.com']
        .includes(host)                                                     ? 'outlook'
    : null;

  if (!CLIENT) return;

  // ── Helpers ───────────────────────────────────────────────────────────────
  function debounce(fn, ms) {
    let timer;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // ── Gmail: extract email content ──────────────────────────────────────────
  function extractGmail() {
    // Try several body selectors for resilience across Gmail versions
    const body =
      document.querySelector('.a3s.aiL') ||
      document.querySelector('.ii.gt .a3s') ||
      document.querySelector('div[data-message-id] .a3s');
    if (!body) return null;

    const subject = document.querySelector('h2.hP')?.textContent?.trim() ?? '';

    const fromEl = document.querySelector('.gD');
    const from   = fromEl
      ? `${fromEl.textContent?.trim()} <${fromEl.getAttribute('email') ?? ''}>`
      : document.querySelector('.go')?.textContent?.trim() ?? '';

    return { body, subject, from, text: body.innerText?.trim() ?? '' };
  }

  // ── Outlook Web: extract email content ────────────────────────────────────
  // Outlook's class names are obfuscated and change; ARIA attributes are stable.
  function extractOutlook() {
    const pane = document.querySelector('[role="main"]');
    if (!pane) return null;

    // Reading pane body — try stable ARIA label first, then class-based fallbacks
    const body =
      pane.querySelector('[aria-label="Message body"]') ||
      pane.querySelector('[class*="scrollBody"] .allowTextSelection') ||
      pane.querySelector('[class*="ReadingPane"] [class*="body"]') ||
      pane.querySelector('[class*="ItemBody"]');
    if (!body) return null;

    const subject =
      pane.querySelector('[class*="subject"]')?.textContent?.trim() ||
      pane.querySelector('h1, h2')?.textContent?.trim() ||
      '';

    const from =
      pane.querySelector('[class*="sender"] [class*="name"]')?.textContent?.trim() ||
      pane.querySelector('[aria-label*="From"]')?.textContent?.trim() ||
      '';

    return { body, subject, from, text: body.innerText?.trim() ?? '' };
  }

  // ── Extract from whichever client is active ───────────────────────────────
  function extractEmail() {
    return CLIENT === 'gmail' ? extractGmail() : extractOutlook();
  }

  // ── Badge: loading state ──────────────────────────────────────────────────
  function showLoading(anchorEl) {
    removeBadge();
    const el = document.createElement('div');
    el.id        = BADGE_ID;
    el.className = 'pd-badge pd-loading';
    el.innerHTML = `
      <div class="pd-row">
        <span class="pd-spinner"></span>
        <span class="pd-status-text">Scanning for phishing…</span>
      </div>`;
    anchorEl.parentElement?.insertBefore(el, anchorEl);
  }

  // ── Badge: result state ───────────────────────────────────────────────────
  function showResult(anchorEl, data) {
    removeBadge();
    const { risk_level: risk, confidence, prediction, plain_english_summary: summary } = data;
    const pct = prediction === 'phishing'
      ? (confidence * 100).toFixed(0)
      : ((1 - confidence) * 100).toFixed(0);

    const el = document.createElement('div');
    el.id        = BADGE_ID;
    el.className = `pd-badge pd-${risk.toLowerCase()}`;
    el.innerHTML = `
      <div class="pd-row">
        <span class="pd-pill">${escHtml(risk)}</span>
        <span class="pd-conf">${escHtml(pct)}% confidence</span>
        <button class="pd-why-btn" aria-expanded="false">Why?</button>
      </div>
      <div class="pd-summary" hidden>${escHtml(summary ?? '')}</div>`;

    el.querySelector('.pd-why-btn').addEventListener('click', function () {
      const expanded = this.getAttribute('aria-expanded') === 'true';
      this.setAttribute('aria-expanded', String(!expanded));
      this.textContent = expanded ? 'Why?' : 'Hide';
      el.querySelector('.pd-summary').hidden = expanded;
    });

    anchorEl.parentElement?.insertBefore(el, anchorEl);
  }

  // ── Badge: error state ────────────────────────────────────────────────────
  function showError(anchorEl, msg) {
    removeBadge();
    const el = document.createElement('div');
    el.id        = BADGE_ID;
    el.className = 'pd-badge pd-error';
    el.innerHTML = `<div class="pd-row"><span class="pd-status-text">PhishingDetector: ${escHtml(msg)}</span></div>`;
    anchorEl.parentElement?.insertBefore(el, anchorEl);
  }

  function removeBadge() {
    document.getElementById(BADGE_ID)?.remove();
  }

  // ── API call ──────────────────────────────────────────────────────────────
  async function callApi(emailText) {
    const { apiUrl = DEFAULT_API } = await chrome.storage.sync.get('apiUrl');
    const base = apiUrl.replace(/\/$/, '');

    const res = await fetch(`${base}/classify`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ email_text: emailText }),
    });

    if (!res.ok) throw new Error(`API returned ${res.status}`);
    return res.json();
  }

  // ── Main scan logic ───────────────────────────────────────────────────────
  async function scanEmail() {
    const extracted = extractEmail();
    if (!extracted) return;

    const { body, subject, from, text } = extracted;
    if (!text) return;

    // Use a fingerprint of the content to avoid re-scanning the same email
    const fingerprint = text.slice(0, 120);
    if (body.dataset[SCAN_KEY] === fingerprint) return;
    body.dataset[SCAN_KEY] = fingerprint;

    // Build a minimal email with headers so the model has full context
    const fullText = [
      from    && `From: ${from}`,
      subject && `Subject: ${subject}`,
      '',
      text,
    ].filter(Boolean).join('\n');

    showLoading(body);

    try {
      const data = await callApi(fullText);
      showResult(body, data);
    } catch (err) {
      console.error('[PhishingDetector]', err);
      showError(body, 'Could not reach API. Check extension settings.');
    }
  }

  const debouncedScan = debounce(scanEmail, 700);

  // Watch for DOM changes (email opens/switches)
  new MutationObserver(debouncedScan)
    .observe(document.body, { childList: true, subtree: true });

  // Gmail navigates via URL hash; Outlook via pushState
  window.addEventListener('hashchange',   debouncedScan);
  window.addEventListener('popstate',     debouncedScan);

  // Initial scan in case an email is already open on load
  debouncedScan();

})();
