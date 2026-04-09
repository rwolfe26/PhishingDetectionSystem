# Deployment Guide

This app runs on **Render** (Docker) and loads model files from **Hugging Face Hub** at startup.
Follow the three steps below once, then every future `git push` auto-deploys.

---

## Step 1 — Upload models to Hugging Face Hub

### 1a. Create a free Hugging Face account
Go to [huggingface.co/join](https://huggingface.co/join) if you don't already have one.

### 1b. Create a new model repository
1. Click your profile icon → **New Model**
2. Name it `phishing-detector`
3. Set visibility to **Public**
4. Click **Create Model**

### 1c. Get an access token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → name it `upload` → Role: **Write**
3. Copy the token (starts with `hf_...`)

### 1d. Install the HF CLI and upload
```bash
cd ~/Detection_System
source .venv/bin/activate
pip install huggingface_hub

# Log in (paste your token when prompted)
huggingface-cli login

# Upload all three model files (takes a few minutes — lsa_encoder.pkl is 346 MB)
huggingface-cli upload your-hf-username/phishing-detector \
    models/phishing_classifier.pkl \
    models/pipeline_metadata.pkl \
    models/lsa_encoder.pkl \
    --repo-type model
```

When it finishes, visit `https://huggingface.co/your-hf-username/phishing-detector`
and confirm all three `.pkl` files are listed.

---

## Step 2 — Deploy to Render

### 2a. Create a free Render account
Go to [render.com](https://render.com) and sign up with GitHub.

### 2b. Create a new Web Service
1. Click **New → Web Service**
2. Connect your **GitHub repo** (`PhishingDetectionSystem`)
3. Render detects the `Dockerfile` automatically — no changes needed
4. Set the following:

| Field | Value |
|-------|-------|
| Name | `phishing-detector` |
| Region | Oregon (US West) |
| Branch | `main` |
| Plan | **Free** (or Starter $7/mo for always-on) |

### 2c. Set environment variables
In the Render dashboard under **Environment**, add:

| Key | Value |
|-----|-------|
| `HF_REPO_ID` | `your-hf-username/phishing-detector` |
| `MODEL_DIR` | `/app/models` |
| `CORS_ORIGINS` | `*` |

> **Note:** If you made the HF repo private, also add `HF_TOKEN` with a read-only token.
> Never commit tokens to git — set them only in the Render dashboard.

### 2d. Deploy
Click **Create Web Service**.

Render will:
1. Pull your GitHub repo
2. Build the Docker image (`pip install -r requirements.txt`)
3. Run `python download_models.py` — downloads the 3 model files from HF Hub (~1 min)
4. Start `uvicorn api.main:app`

**First deploy takes ~5 minutes** (model download). Subsequent deploys are faster
because Render caches the disk between deploys.

### 2e. Confirm it's live
Visit `https://your-app-name.onrender.com/health` — you should see:
```json
{"status": "ok", "model_loaded": true}
```

Then go to `https://your-app-name.onrender.com` to use the classifier.

---

## Free tier limitations

| Limitation | Detail |
|------------|--------|
| Spins down after 15 min of inactivity | First visitor waits ~30s for cold start |
| 512 MB RAM | Enough for inference; tight during model load |
| Shared CPU | Slower than paid, fine for demos |

To avoid cold starts, upgrade to **Starter ($7/mo)** for always-on hosting.

---

## Updating the deployed app

```bash
# Push any code change — Render auto-deploys on push to main
git push origin main
```

To update the model files:
```bash
huggingface-cli upload your-hf-username/phishing-detector \
    models/phishing_classifier.pkl \
    models/pipeline_metadata.pkl \
    models/lsa_encoder.pkl \
    --repo-type model
```
Then trigger a manual redeploy in the Render dashboard (or push a commit).

---

## Local Docker test (before deploying)

```bash
docker build -t phishing-detector .
docker run -p 8000:8000 \
  -e HF_REPO_ID=your-hf-username/phishing-detector \
  -e MODEL_DIR=/app/models \
  phishing-detector
```
