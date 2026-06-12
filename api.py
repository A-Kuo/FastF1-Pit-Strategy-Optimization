#!/usr/bin/env python3
"""
FastAPI Health Check & Prediction Endpoints
============================================

Provides liveness/readiness checks for Kubernetes orchestration and
basic prediction endpoints for integration with external systems.

Run standalone:
    python api.py

Or with Streamlit:
    docker-compose up

Endpoints:
  GET  /health      - Liveness check (is service running?)
  GET  /ready       - Readiness check (can we serve predictions?)
  POST /predict     - Single prediction
  POST /batch-predict - Batch predictions
"""

import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import logging

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="F1 Pit Strategy API",
    description="Real-time pit stop prediction for Formula 1 strategy optimization",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE (loaded on startup)
# ─────────────────────────────────────────────────────────────

MODEL = None
SCALER = None
METRICS = None
FEATURE_COLS = ["DegradationRate", "StintAgeSquared", "RaceProgress", "PaceDelta"]


@app.on_event("startup")
async def load_models():
    """Load trained model and scaler on startup."""
    global MODEL, SCALER, METRICS

    try:
        # Load XGBoost model
        with open("models/xgboost_model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        logger.info("✓ Loaded XGBoost model")

        # Load scaler
        with open("models/scaler.pkl", "rb") as f:
            SCALER = pickle.load(f)
        logger.info("✓ Loaded StandardScaler")

        # Load metrics
        with open("models/metrics.pkl", "rb") as f:
            METRICS = pickle.load(f)
        logger.info(f"✓ Loaded metrics (ROC-AUC {METRICS['roc_auc']:.4f})")

        logger.info("Ready to serve predictions")

    except FileNotFoundError as e:
        logger.warning(f"Model artifacts not found during startup: {e}")
        logger.warning(
            "Model artifacts are missing; readiness and prediction endpoints will "
            "return 503 until models are successfully loaded."
        )


# ─────────────────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Liveness check response."""
    status: str = Field(..., description="Service status: 'healthy'")
    timestamp: str = Field(..., description="ISO timestamp")
    version: str = Field(..., description="API version")


class ReadyResponse(BaseModel):
    """Readiness check response."""
    status: str = Field(..., description="Service readiness: 'ready' or 'not_ready'")
    message: str = Field(..., description="Details")
    model_info: Dict[str, Any] = Field(default=None, description="Model metadata")


class PredictionRequest(BaseModel):
    """Single prediction request."""
    degradation_rate: float = Field(..., ge=-0.5, le=0.5, description="OLS slope of lap time vs stint age")
    stint_age_squared: float = Field(..., ge=0, le=3600, description="Tyre age squared")
    race_progress: float = Field(..., ge=0, le=1, description="Lap / max_lap (0-1)")
    pace_delta: float = Field(..., ge=-3, le=3, description="Driver lap time vs 5-lap rolling median")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    pit_probability: float = Field(..., ge=0, le=1, description="Probability of pit within 5 laps")
    pit_decision: str = Field(..., description="Decision: PIT_NOW, MONITOR, or STAY_OUT")
    decision_threshold: float = Field(..., description="Configured threshold τ")
    timestamp: str = Field(..., description="ISO timestamp")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    features: List[List[float]] = Field(..., description="Array of [deg_rate, stint_age², race_progress, pace_delta]")
    threshold: float = Field(default=0.60, ge=0.1, le=0.9, description="Decision threshold")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    pit_probabilities: List[float]
    pit_decisions: List[str]
    timestamp: str


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Liveness check.

    Returns immediately if the service is running. Used by Kubernetes for
    liveness probes (determines if pod should be restarted).

    **Response:**
    - `status`: Always "healthy" if this endpoint responds
    - `timestamp`: Current UTC time
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.get("/ready", response_model=ReadyResponse, tags=["health"])
async def readiness_check() -> ReadyResponse:
    """
    Readiness check.

    Verifies that required models and scalers are loaded and functional.
    Used by Kubernetes for readiness probes (determines if pod should
    receive traffic).

    **Response:**
    - `status`: "ready" if all models loaded, "not_ready" otherwise
    - `model_info`: Model metadata (ROC-AUC, F1, feature names)
    """
    if MODEL is None or SCALER is None or METRICS is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Service is initializing."
        )

    return ReadyResponse(
        status="ready",
        message="All models loaded and ready to serve predictions",
        model_info={
            "model_type": "XGBoost",
            "roc_auc": float(METRICS["roc_auc"]),
            "f1_score": float(METRICS["f1"]),
            "recall": float(METRICS["recall"]),
            "threshold": float(METRICS["threshold"]),
            "features": FEATURE_COLS,
            "train_size": int(METRICS["train_size"]),
            "test_size": int(METRICS["test_size"]),
        }
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Single pit probability prediction.

    Takes lap-level features and returns probability of pit stop within
    next 5 laps, along with binary decision at configured threshold (τ=0.60).

    **Request:**
    - `degradation_rate`: OLS slope of lap time vs tyre life per stint (s/lap)
    - `stint_age_squared`: Tyre age squared (laps²)
    - `race_progress`: Current lap / total laps (0-1 scale)
    - `pace_delta`: Driver lap time minus 5-lap rolling median (seconds)

    **Response:**
    - `pit_probability`: Float [0, 1]
    - `pit_decision`: "PIT_NOW" (prob ≥ τ), "MONITOR", or "STAY_OUT"
    - `decision_threshold`: Configured threshold (0.60)
    """
    if MODEL is None or SCALER is None or METRICS is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Convert to feature array in correct order
        X = np.array([
            request.degradation_rate,
            request.stint_age_squared,
            request.race_progress,
            request.pace_delta
        ]).reshape(1, -1)

        # Scale
        X_scaled = SCALER.transform(X)

        # Predict
        pit_probability = float(MODEL.predict_proba(X_scaled)[0, 1])

        # Decision
        threshold = METRICS.get("threshold", 0.60)
        if pit_probability >= threshold:
            decision = "PIT_NOW"
        elif pit_probability >= threshold - 0.1:
            decision = "MONITOR"
        else:
            decision = "STAY_OUT"

        logger.info(f"Prediction: {pit_probability:.3f} → {decision}")

        return PredictionResponse(
            pit_probability=pit_probability,
            pit_decision=decision,
            decision_threshold=threshold,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["inference"])
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch pit probability predictions.

    Takes multiple lap samples and returns probabilities + decisions.

    **Request:**
    - `features`: List of [deg_rate, stint_age², race_progress, pace_delta] arrays
    - `threshold`: Decision threshold (optional, default 0.60)

    **Response:**
    - `pit_probabilities`: List of probabilities [0, 1]
    - `pit_decisions`: List of decisions ("PIT_NOW", "MONITOR", "STAY_OUT")
    """
    if MODEL is None or SCALER is None or METRICS is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        if not request.features:
            raise ValueError("features cannot be empty")

        X = np.array(request.features, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != 4:
            raise ValueError(f"Expected 2D array with 4 features, got shape {X.shape}")

        # Scale
        X_scaled = SCALER.transform(X)

        # Predict
        pit_probs = MODEL.predict_proba(X_scaled)[:, 1]

        # Decisions
        threshold = request.threshold
        decisions = [
            "PIT_NOW" if p >= threshold else ("MONITOR" if p >= threshold - 0.1 else "STAY_OUT")
            for p in pit_probs
        ]

        logger.info(f"Batch prediction: {len(pit_probs)} samples")

        return BatchPredictionResponse(
            pit_probabilities=pit_probs.tolist(),
            pit_decisions=decisions,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/", tags=["info"])
async def root():
    """API documentation and links."""
    return {
        "service": "F1 Pit Strategy Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "ready": "GET /ready",
            "predict": "POST /predict",
            "batch": "POST /batch-predict"
        },
        "model": {
            "type": "XGBoost",
            "features": FEATURE_COLS,
            "threshold": 0.60 if METRICS else None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
