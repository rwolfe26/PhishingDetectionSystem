FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (models are excluded via .dockerignore / .gitignore)
COPY . .

# Models are not baked into the image — mount them at runtime via a volume:
#   docker run -v $(pwd)/models:/app/models ...
# Or train inside the container:
#   docker run ... python run_pipeline.py --train

ENV MODEL_DIR=/app/models
ENV CORS_ORIGINS=*

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
