"""
Phishing Detection API

FastAPI backend for the email phishing detection system.
Exposes endpoints for classification, explanation, and health checks.

Usage:
    uvicorn api.main:app --reload
    python api/main.py  (for quick testing)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import EmailPhishingPipeline, Predictor
from preprocessing import preprocess_email_with_lsa

from .explainer import explain_prediction, explain_with_shap, highlight_phishing_indicators

logger = logging.getLogger(__name__)

# ── Models / state ──────────────────────────────────────────────────────────

_pipeline: Optional[EmailPhishingPipeline] = None
_predictor: Optional[Predictor] = None
_model_dir = Path(__file__).resolve().parent.parent / 'models'


def _load_pipeline():
    global _pipeline, _predictor
    if not _model_dir.exists():
        logger.error("models/ directory not found. Run: python run_pipeline.py --train")
        return False
    try:
        _pipeline = EmailPhishingPipeline()
        _pipeline.load_models(_model_dir)
        _predictor = Predictor()
        logger.info("Pipeline loaded successfully")
        return True
    except Exception as exc:
        logger.error(f"Failed to load pipeline: {exc}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = _load_pipeline()
    if not loaded:
        logger.warning("Running without loaded models — /classify will return 503")
    yield


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Phishing Detection API",
    description="Classify emails as phishing or benign using ML.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the frontend)
static_dir = Path(__file__).parent / 'static'
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Request / Response models ────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    email_text: str
    use_shap: bool = False


class FeatureExplanation(BaseModel):
    name: str
    description: str
    value: float
    importance: float
    direction: str


class ClassifyResponse(BaseModel):
    prediction: str          # "phishing" | "benign"
    confidence: float        # 0.0 – 1.0
    risk_level: str          # "HIGH" | "MEDIUM" | "LOW" | "SAFE"
    top_features: list       # List of FeatureExplanation dicts
    indicators: dict         # Highlighted phishing phrases
    model_loaded: bool


# ── Helpers ──────────────────────────────────────────────────────────────────

def _risk_level(confidence: float, prediction: str) -> str:
    if prediction == 'phishing':
        if confidence >= 0.85:
            return 'HIGH'
        if confidence >= 0.60:
            return 'MEDIUM'
        return 'LOW'
    return 'SAFE'


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    html_path = static_dir / 'index.html'
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return HTMLResponse("<h1>Phishing Detector API</h1><p>See <a href='/docs'>/docs</a></p>")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None and _pipeline.classifier is not None,
        "model_dir": str(_model_dir),
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify_email(request: ClassifyRequest):
    """
    Classify an email as phishing or benign.

    Returns prediction, confidence, risk level, top contributing features,
    and highlighted phishing indicators.
    """
    if _pipeline is None or _pipeline.classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python run_pipeline.py --train' first."
        )

    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="email_text must not be empty")

    try:
        result = preprocess_email_with_lsa(request.email_text, _pipeline.lsa_encoder)
        feature_vector = result['combined_vector']

        X = feature_vector.reshape(1, -1)
        prediction_int = _pipeline.classifier.predict(X)[0]
        proba = _pipeline.classifier.predict_proba(X)[0]
        confidence = float(proba[1])  # probability of being phishing

        prediction_str = 'phishing' if prediction_int == 1 else 'benign'
        risk = _risk_level(confidence, prediction_str)

        # Explainability
        if request.use_shap:
            top_features = explain_with_shap(_pipeline, feature_vector) or \
                           explain_prediction(_pipeline, feature_vector)
        else:
            top_features = explain_prediction(_pipeline, feature_vector)

        indicators = highlight_phishing_indicators(request.email_text, feature_vector)

        return ClassifyResponse(
            prediction=prediction_str,
            confidence=round(confidence, 4),
            risk_level=risk,
            top_features=top_features,
            indicators=indicators,
            model_loaded=True,
        )

    except Exception as exc:
        logger.exception(f"Error during classification: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/classify/file")
async def classify_file(file: UploadFile = File(...), use_shap: bool = False):
    """
    Classify an uploaded .eml or plain text email file.
    """
    content = await file.read()
    email_text = content.decode('utf-8', errors='ignore')

    from pydantic import BaseModel as BM
    req = ClassifyRequest(email_text=email_text, use_shap=use_shap)
    return await classify_email(req)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
