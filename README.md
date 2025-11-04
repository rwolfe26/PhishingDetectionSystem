# PhishingDetectionSystem

**Product Features (User-Facing)**

Email input:
    Paste raw email text (body only)
    Upload .eml/.msg (preserve headers)
    Optional, depending on our path: Gmail/IMAP read-only connector (OAuth)

Classification & confidence:
    Label: Phishing / Suspicious / Legitimate
    Confidence score (well-calibrated; see calibration below)

Explanation:
    Why flagged? Highlight phrases/URLs/headers that contributed most
    Indicator badges: Urgency, Credential harvest, Spoofed display name, Link-domain mismatch, Attachment risk, Brand impersonation..etc
    "Show details" panels with header analysis, URL expansions, and model feature contributions 

Safe link/attatchment preview:
    Resolve redirects safely (no active fetch of payloads; HEAD/metadata only)

User actions:
    Mark as phishing/ not phishing (feedback to model)
    One-click generate saftye tips for the flagged indiciators

Accessibility & UX:
    Copy-safe summary (plain language, 1-2 sentences)


ML Design Requirements

Datasets:
    Training: public datasets (like collections of curated phishing datasets) augmented with synthetic phish variants

Feature Engineering (classic + NLP):
    Header: From/Reply-To/domain match, DKIM,SPF/DMARC results (if available)
    URLS: Count, entropy, length, TLD rarity, homograph patterns, domain age, redirect chain length, mismatch between anchor text and href
    Body (lexical/strucutal): Urgency words, financial asks, credential phrases, spelling patterns, HTML forms, base64 blobs, hidden texts, CSS tricks, unicode confusables
    NLP: Transformer embeddings over subject + body; sentence-level persusasion cues
    Attatchments: Types, macro flags, passsword-protected heuristics 

Models:
    MVP: Linear/Tree ensemble (LogReg/XGBoost) on engineered features + pooled transformer embeddings
    v1.0: Lightweight transformer fine-tuned on email text with multu-task heads (classification + cue tagging)
    Calibration: Platt/Isotonic on validation for trustworthy confidence
    Thresholding: Seperate thresholds for "Phishing" vs "Suspicious" tuned to target recall on phish (e.g, ≥ 0.95)

Explainability:
    Feature attributions: SHAP/Integrated Gradients for top signals; highlight spans in text and specific URLs/ headers
    Rules overlay: Human-readable checks ("Display name does not match domain", "Link text != href domain")

Robustness:
    Adverisal text tricks (zero-width spaces, homoglyphs, HTML obsfuscation)
    URL shortners and multi-redirect chains
    Template diversity: brand impersonation, payroll/2FA, package delivery, financial refund


**System & Architecture**

App form factor:
    MVP: Web app (FastAPI/Flash + React)
    Alt: Local Desktop (PyQT/TK) if avoiding OAuth complexity

Pipelines:
    Ingestion -> MIME parse -> header/body/URL extraction -> preprocess (tokenize, normalize, defang links) -> featureization/embedding -> model -> explanations -> report

Integrations:
    Optional Gmail/IMAP connector (read-only, least-privledge scopes, no deletions)
    DNS/WHOIS domain age lookup (cache & rate-limit; can be deferred)

Performance targets:
    Latency: <800 ms for paste-inference; <2s with URL expansion
    Memory: model ≤ 200MB (cpu friendly)
    Concurrency: 10-50 RPS dev target with batching disabled

Observability:
    Structued logs (no raw PII), model inputs hashed, decision trace ID
    Feature drift monitoring 

CI/ML ops (lightweight)
    Reproducable training scripts, experiment tracking, model registry, semantic versioning 
    Unit tests for parsers, URL resolver, feature builders, and decision thresholds


**Security, Privacy, and Compliance**

PII handling:
    Default: client-side inference or server: process-in-memory, do not persist raw emails
    If storing samples for improvement, explicit consent + irreversible redaction (names, emails, IDs, phone, addresses)

Secrets & auth:
    OAuth with least-privledge scopes; rotate client secrets, short-lived tokens, CSRF protection 

Content safety:
    Never auto-fetch attachments, only metadata
    URL expansion via safe HEAD/GET with disabled JS; block active content

Abuse & misuse
    Rate limiting: Prevent tool being used to validate phishing quality (no "tips to bypass")

Disclosure:
    Clear statement: model is assistive, false negatives possible, saftey guidance included

Human-in-the-loop:
    Reviewer dashboard for disagree cases, active-learning queue from "user-flagged" items

Release criteria:
    Ready to release when: metrics hit targets on validation + robustness; calibration ECE < 0.05, explanations pass clarity rubric


**IF WE ARE PLANNING ON HOSTING A USER STUDY**

Design:
    Between-subjects: Control (no explanations) vs Explainable UI
    Tasks: classify 12-16 emails mixed phish/ham; capture time-to-decision and confidence

Measures:
    Accuracy, sensitivity (phish recall), decision time, post-task comprehension (what cues they learned), SUS or UEQ score
    Pre/post quiz on phishing cues to measure educational impact

Consent & Ethics:
    IRB/departmental approval template, anonymized logging; debrief with saftey tips

Analysis:
    Hypothesis: Explanations improve correctness and cue recall without large time cost


**Requirements Traceability**

MVP (x weeks)
    Paste email -> classify -> calibrated confidence -> top-3 cues highlighted
    Basic URL defang + redirect resolution (metadata only)
    Offline model, no data rentention, exportable JSON report

v1.0 (x weeks)
    OAuth gmail/IMAP ingestion (real-only)
    Rule + model hybrid with SHAP spans; attachment triage, drift monitor
    Admin page for mislabeled queue, active learning, and A/B explainability variants

    


