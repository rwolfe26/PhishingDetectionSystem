# Documentation Index

This directory contains detailed documentation for the Email Phishing Detection System.

## 📚 Available Documentation

### [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md)
**Modular Architecture Guide**

Detailed documentation of the refactored modular structure:
- Module descriptions and responsibilities
- File organization
- Usage examples for each module
- How to extend and add new features
- Benefits of the modular design

**Read this if you want to:**
- Understand the codebase architecture
- Add new features or modules
- Contribute to the project
- Integrate components in your own code

---

### [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
**Complete Pipeline Usage Guide**

Comprehensive guide to using the complete pipeline:
- Installation instructions
- Training workflow (including `--include-feedback` for active learning)
- Prediction workflow
- Feature breakdown (47 numeric + 128 LSA = 175 total features)
- IMAP monitor CLI reference
- API endpoint reference (classifier, dashboard, feedback, monitor)
- Performance tips and troubleshooting
- Advanced usage patterns

**Read this if you want to:**
- Learn how to use the pipeline end-to-end
- Train models on your own data
- Understand feature engineering
- Optimize performance
- Deploy the system

---

### [LSA_PREPROCESSING_INTEGRATION.md](LSA_PREPROCESSING_INTEGRATION.md)
**LSA and Preprocessing Integration**

Technical documentation on how LSA semantic analysis integrates with preprocessing:
- Architecture overview
- Integration functions (`fit_lsa_encoder`, `preprocess_email_with_lsa`)
- Combined feature vector composition (47 numeric + 128 LSA = 175 total)
- Usage examples and code snippets
- Troubleshooting LSA issues

**Read this if you want to:**
- Understand how LSA and preprocessing connect
- Use the integrated functions in your code
- Debug LSA-related issues
- Customize LSA parameters
- Understand semantic feature extraction

---

### [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
**Project Refactoring Summary**

Documentation of the transition from monolithic to modular structure:
- Before/after comparison
- Module breakdown
- Testing results
- Migration guide from old code
- Verification checklist

**Read this if you want to:**
- Understand the project evolution
- Migrate from old `pipeline.py` code
- Learn about design decisions
- See what changed and why

---

## 🗂️ Documentation by Use Case

### For New Users
1. Start with the main [README.md](../README.md) in the root directory
2. Follow the Quick Start guide
3. Read [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for detailed usage

### For Developers
1. Read [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md) for architecture
2. Review [LSA_PREPROCESSING_INTEGRATION.md](LSA_PREPROCESSING_INTEGRATION.md) for integration details
3. Check [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for project history

### For Contributors
1. Read [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md) to understand the codebase
2. Review coding patterns in existing modules
3. Follow the structure when adding new features
4. Write tests for new code

### For Researchers
1. Review [LSA_PREPROCESSING_INTEGRATION.md](LSA_PREPROCESSING_INTEGRATION.md) for methodology
2. Check `../bert_base/lsa_research_report.md` for LSA research notes
3. Examine feature extraction in [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)

---

## 📖 Additional Resources

### In Other Directories

- **`../bert_base/lsa_research_report.md`** - LSA methodology and research
- **`../DEPLOY.md`** - Step-by-step deployment guide (Hugging Face Hub + Render / HF Spaces)
- **`../examples/`** - Example scripts demonstrating usage
- **`../tests/`** - 60+ pytest tests covering pipeline, API, monitor, and feedback

### External Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/) - For machine learning APIs
- [SpamAssassin](https://spamassassin.apache.org/) - Dataset source
- [RFC 822](https://www.ietf.org/rfc/rfc822.txt) - Email format specification

---

## 🔍 Quick Reference

### Common Tasks

| Task | Documentation |
|------|---------------|
| Install and run | [Main README](../README.md) → Running locally |
| Train a model | [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) → Training |
| Retrain with feedback | [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) → `--include-feedback` |
| Make predictions | [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) → Prediction |
| Deploy to production | [DEPLOY.md](../DEPLOY.md) |
| Understand architecture | [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md) |
| Add new features | [MODULAR_STRUCTURE.md](MODULAR_STRUCTURE.md) → Extending |
| Use in own code | [LSA_PREPROCESSING_INTEGRATION.md](LSA_PREPROCESSING_INTEGRATION.md) → Examples |
| Troubleshoot issues | [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) → Troubleshooting |

---

**Last Updated**: April 2026
