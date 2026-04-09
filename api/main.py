"""
Phishing Detection API

FastAPI backend for the email phishing detection system.
Exposes endpoints for classification, explanation, and health checks.

Usage:
    uvicorn api.main:app --reload
    python api/main.py  (for quick testing)

Environment variables:
    CORS_ORIGINS  Comma-separated list of allowed origins (default: "*")
    MODEL_DIR     Path to directory containing trained model files
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from pipeline import EmailPhishingPipeline
from preprocessing import preprocess_email_with_lsa

from .explainer import explain_prediction, explain_with_shap, highlight_phishing_indicators


# ── Structured JSON logging ──────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'time': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if record.exc_info:
            payload['exc_info'] = self.formatException(record.exc_info)
        extra_keys = {'request_id', 'path', 'method', 'status_code', 'duration_ms'}
        for key in extra_keys:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload)


def _configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)


# ── Request-ID middleware ────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a UUID request-ID to every request and log completion."""

    async def dispatch(self, request: Request, call_next):
        import time
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers['X-Request-ID'] = request_id
        logger.info(
            'request',
            extra={
                'request_id': request_id,
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
                'duration_ms': duration_ms,
            },
        )
        return response


# ── Config ───────────────────────────────────────────────────────────────────

MAX_EMAIL_BYTES = 1 * 1024 * 1024  # 1 MB hard limit for email text

_cors_origins_raw = os.environ.get('CORS_ORIGINS', '*')
_cors_origins = [o.strip() for o in _cors_origins_raw.split(',') if o.strip()] or ['*']

_model_dir = Path(
    os.environ.get('MODEL_DIR', str(Path(__file__).resolve().parent.parent / 'models'))
)

_monitor_db = Path(
    os.environ.get('MONITOR_DB', str(Path(__file__).resolve().parent.parent / 'monitor.db'))
)

# ── Models / state ──────────────────────────────────────────────────────────

_pipeline: Optional[EmailPhishingPipeline] = None


def _load_pipeline():
    global _pipeline
    if not _model_dir.exists():
        logger.error('models/ directory not found — run: python run_pipeline.py --train')
        return False
    try:
        _pipeline = EmailPhishingPipeline()
        _pipeline.load_models(_model_dir)
        logger.info('pipeline loaded successfully', extra={'model_dir': str(_model_dir)})
        return True
    except Exception as exc:
        logger.error('failed to load pipeline', extra={'error': str(exc)})
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = _load_pipeline()
    if not loaded:
        logger.warning('running without loaded models — /classify will return 503')
    yield


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Phishing Detection API",
    description="Classify emails as phishing or benign using ML.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the monitoring dashboard HTML page."""
    html_path = static_dir / 'dashboard.html'
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return HTMLResponse("<h1>Dashboard</h1><p>dashboard.html not found.</p>")


@app.get("/api/monitor/stats")
async def monitor_stats():
    """Aggregate classification counts from the monitor database."""
    if not _monitor_db.exists():
        return {"total": 0, "phishing": 0, "benign": 0, "db_exists": False}
    try:
        from email_monitor.storage import ClassificationStore
        store = ClassificationStore(str(_monitor_db))
        stats = store.stats()
        stats["db_exists"] = True
        return stats
    except Exception as exc:
        logger.exception("monitor_stats error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/monitor/history")
async def monitor_history(limit: int = 50):
    """Recent classified emails from the monitor database."""
    if not _monitor_db.exists():
        return {"items": [], "db_exists": False}
    try:
        from email_monitor.storage import ClassificationStore
        store = ClassificationStore(str(_monitor_db))
        items = store.recent(limit=min(limit, 200))
        return {"items": items, "db_exists": True}
    except Exception as exc:
        logger.exception("monitor_history error")
        raise HTTPException(status_code=500, detail=str(exc))


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

    if len(request.email_text.encode('utf-8')) > MAX_EMAIL_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Email exceeds maximum size of {MAX_EMAIL_BYTES // 1024} KB"
        )

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

    except HTTPException:
        raise
    except Exception:
        logger.exception('classification error')
        raise HTTPException(status_code=500, detail="Internal classification error")


@app.post("/classify/file")
async def classify_file(file: UploadFile = File(...), use_shap: bool = False):
    """
    Classify an uploaded .eml or plain text email file.
    """
    content = await file.read()
    if len(content) > MAX_EMAIL_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_EMAIL_BYTES // 1024} KB"
        )
    email_text = content.decode('utf-8', errors='ignore')
    req = ClassifyRequest(email_text=email_text, use_shap=use_shap)
    return await classify_email(req)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
