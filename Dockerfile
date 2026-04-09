FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (models are excluded via .dockerignore / .gitignore)
COPY . .

# Models are downloaded from Hugging Face Hub at container startup.
# Set HF_REPO_ID to your repo (e.g. "rwolfe26/phishing-detector").
# For local dev, mount models instead:
#   docker run -v $(pwd)/models:/app/models -e MODEL_DIR=/app/models ...

ENV MODEL_DIR=/app/models
ENV CORS_ORIGINS=*
# HF_REPO_ID and HF_TOKEN are injected by Render (or docker run -e)

EXPOSE 8000

# Download models from HF Hub (no-op if already present or HF_REPO_ID unset),
# then start the API server.
CMD ["sh", "-c", "python download_models.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
