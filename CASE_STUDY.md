# Case Study: Email Phishing Detection System

**Ryan Wolfe · Skidmore College · Senior Research Project**

---

## Background

Phishing remains one of the most prevalent and costly attack vectors in cybersecurity — responsible for over 90% of data breaches according to industry reports. Despite this, most detection tooling is either locked behind enterprise security platforms or limited to simple rule-based spam filters that are trivially bypassed by modern phishing campaigns.

For my senior research project I wanted to build something that addressed this gap directly: a machine learning system that could detect phishing emails with high accuracy, explain its reasoning in plain English, and be accessible outside of a corporate security stack. The goal was to go end-to-end — from raw data and feature engineering through to a deployed, usable product.

---

## The Problem

Phishing emails are deliberately designed to evade detection. Modern campaigns use legitimate-looking domains, avoid known blacklisted URLs, and craft language that mimics real corporate communications. A model that looks only at keywords or known-bad domains will miss the majority of targeted attacks.

The core technical challenge was building a feature set rich enough to catch sophisticated phishing while keeping false positives low enough to be usable in a real inbox. Getting that balance wrong in either direction has real consequences — miss a phishing email and a user gets compromised; flag too many legitimate emails and users stop trusting the tool entirely.

---

## What I Built

The system is composed of several layers that each add a distinct capability:

**Feature engineering** — Rather than relying on a pre-trained model out of the box, I built a 44-feature extractor that captures signals specific to email security: URL structure analysis (IP-based URLs, HTTPS ratio, shortener detection, redirect chain resolution), header anomalies (Reply-To mismatches, SPF/DKIM authentication failures, display-name spoofing), HTML structure (hidden forms, iframes, invisible text), and behavioural patterns (urgency keyword density, brand impersonation scoring via Levenshtein distance, homograph character detection). These 44 numeric features are combined with 128-dimensional LSA semantic embeddings to produce a 175-feature vector per email.

**Machine learning pipeline** — I benchmarked five classifiers — Random Forest, XGBoost, Logistic Regression, Linear SVM, and Gradient Boosting — against the same feature set using 5-fold cross-validation. Random Forest performed best on the combination of accuracy, AUC-ROC, and false positive rate that mattered for this use case, and offered the interpretability advantage of native feature importances.

**Explainability layer** — Every classification produces three levels of explanation: ranked feature importances showing which signals drove the decision, SHAP values for per-prediction Shapley explanations, and a plain-English narrative summary generated from the underlying signals. The plain-English layer was a deliberate product decision — a risk score without context isn't actionable for a non-technical user.

**Production system** — The model is served via a FastAPI backend with structured logging and request-ID middleware, deployed on Hugging Face Spaces via Docker. A browser extension for Chrome injects real-time risk badges directly into Gmail and Outlook Web as emails are opened. An IMAP monitor classifies incoming mail from a live mailbox and stores results in SQLite. A feedback loop allows users to flag incorrect predictions, with corrections feeding back into retraining.

---

## Key Technical Decisions

**Choosing Random Forest over transformer-based models**

Early in the project I explored using a fine-tuned BERT model as the classification backbone. The appeal was obvious — transformers have strong semantic understanding and would handle novel phishing language well. I moved away from this for two reasons. First, inference latency on a CPU-only deployment (the constraint for a free-tier hosting environment) made it impractical for real-time use in a browser extension. Second, email phishing detection is fundamentally a structured-signal problem — the most reliable indicators are header anomalies, URL properties, and authentication failures, none of which a language model is inherently equipped to reason about without careful prompting. A handcrafted feature extractor combined with a fast ensemble classifier gave better performance on the benchmark datasets and was deployable within the infrastructure constraints.

If I were rebuilding this for an enterprise environment with GPU inference available, I would revisit a hybrid approach: use a transformer for the semantic embedding layer and combine it with the structured feature extractor, rather than treating them as alternatives.

**Building the feedback loop before it was needed**

Most ML projects treat retraining as an afterthought. I built the feedback collection and correction pipeline early, before the model was deployed, because the most valuable signal for improving phishing detection is real-world misclassifications — particularly false positives on legitimate email from the user's own inbox. Having the infrastructure in place means improvements are possible without restarting from scratch.

**Plain-English explanations as a first-class feature**

A common failure mode in security tooling is presenting risk scores without context. A user who sees "HIGH — 89% confidence" and doesn't understand why is no better equipped to make a decision than one who saw no alert at all. The plain-English summary — generated deterministically from the underlying feature signals — was built to bridge that gap. This was a product decision as much as a technical one.

---

## Results

Trained on 17,204 emails across 5 public datasets (SpamAssassin, CEAS 2008, Nazario, Enron, PHIUSIIL):

| Metric | Score |
|--------|-------|
| Accuracy | 98.55% |
| Precision | 98.04% |
| Recall | 97.71% |
| F1-Score | 97.88% |
| AUC-ROC | 99.91% |
| False Positive Rate | 1.02% |

---

## Honest Limitations

**The training data doesn't fully reflect modern inboxes.** The benchmark datasets, while standard in academic research, skew toward older phishing campaigns and a particular distribution of legitimate email (heavy on mailing lists and corporate newsletters). When I tested the browser extension against my own Gmail inbox I observed a higher false positive rate than the benchmark numbers suggest — certain legitimate marketing emails and automated notifications triggered the model. This is a known and important gap.

The path to closing it is threefold: augmenting training data with more recent phishing samples, incorporating user-specific calibration via the feedback loop, and adjusting the confidence threshold at which warnings are shown. The architecture already supports all three — it's a data and tuning problem, not a structural one.

**If I were starting over**, I would spend more time upfront researching the feature sets used by commercial email security products (Proofpoint, Mimecast, Microsoft Defender for Office 365) before designing my own extractor. I built the 44-feature set largely from academic literature and first principles, which worked well, but enterprise systems have years of production signal about which features are most robust to adversarial evasion. Starting from that baseline would have saved iteration time.

---

## What Production Would Require

The current system is a research prototype with production-quality engineering around it. Taking it to real-world deployment at scale would require:

- **Larger, more recent training data** — ideally sourced from live email telemetry with human-verified labels, rather than static academic benchmarks
- **Adversarial robustness testing** — evaluating performance against emails specifically crafted to evade the known feature set
- **Per-user calibration** — adjusting detection thresholds based on individual false positive feedback to reduce alert fatigue
- **Rate limiting and abuse prevention** — the API is currently open; production use would require authentication and per-user quotas
- **A privacy-preserving inference path** — for enterprise use, email content should ideally never leave the user's infrastructure; the model and feature extractor are portable enough to run locally

---

## Takeaways

This project deepened my understanding of the full ML engineering lifecycle — not just model training and evaluation, but deployment, monitoring, explainability, and the feedback loops that keep a system useful over time. It also reinforced something that I think gets underemphasised in academic ML work: the gap between benchmark accuracy and real-world usefulness is almost always a data distribution problem, and closing it requires operational infrastructure as much as modelling work.

The codebase is open source at [github.com/rwolfe26/PhishingDetectionSystem](https://github.com/rwolfe26/PhishingDetectionSystem). A live demo is available at [rwolfe26-phishing-detector.hf.space](https://rwolfe26-phishing-detector.hf.space).
