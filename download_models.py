"""
Model Download Script

Downloads trained model files from Hugging Face Hub if they are not
already present locally.  Called automatically by the Docker container
at startup (see Dockerfile CMD).

Environment variables:
    HF_REPO_ID   Your HF repo, e.g. "rwolfe26/phishing-detector"
                 If unset the script exits silently (local dev mode).
    MODEL_DIR    Where to save the models (default: ./models)
    HF_TOKEN     Optional — required only for private repos
"""

import os
import sys
from pathlib import Path

HF_REPO_ID = os.environ.get("HF_REPO_ID", "").strip()
MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "./models"))
HF_TOKEN   = os.environ.get("HF_TOKEN", "").strip() or None

MODEL_FILES = [
    "phishing_classifier.pkl",
    "pipeline_metadata.pkl",
    "lsa_encoder.pkl",
]


def _all_present() -> bool:
    return all((MODEL_DIR / f).exists() for f in MODEL_FILES)


def download_models() -> None:
    if not HF_REPO_ID:
        print("[download_models] HF_REPO_ID not set — skipping download (local mode).")
        return

    if _all_present():
        print("[download_models] All model files already present — skipping download.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download_models] huggingface_hub not installed — skipping download.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download_models] Downloading models from {HF_REPO_ID} → {MODEL_DIR}")

    for filename in MODEL_FILES:
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"  ✓ {filename} (already exists)")
            continue
        print(f"  ↓ {filename} ...", end="", flush=True)
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(MODEL_DIR),
                repo_type="model",
                token=HF_TOKEN or None,
            )
            print(" done")
        except Exception as exc:
            print(f" FAILED: {exc}")
            sys.exit(1)

    print("[download_models] All models downloaded successfully.")


if __name__ == "__main__":
    download_models()
