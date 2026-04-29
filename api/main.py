import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import predictor
from api.schemas import (
    ForecastPoint,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

APP_VERSION = "1.0.0"
MODEL_NAME = "mlops-timeseries-ensemble"


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="MLOps Time Series Ensemble API",
    description=(
        "Forecast API serving an ensemble of ARIMA, Prophet, LightGBM, and XGBoost "
        "models trained on the ETTh1 (Electricity Transformer Temperature) dataset."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def root():
    return {"message": "MLOps Time Series Ensemble API", "version": APP_VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True, tags=["forecast"])
async def predict(request: PredictRequest):
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run the training pipeline first: `dvc repro`",
        )
    try:
        result = predictor.predict(
            start_datetime=request.start_datetime,
            steps=request.steps,
            include_components=request.include_components,
            history=request.history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        model=MODEL_NAME,
        steps=request.steps,
        forecast=[ForecastPoint(**pt) for pt in result["forecast"]],
        ensemble_weights=result.get("ensemble_weights"),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def get_metrics():
    metrics_path = "metrics/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail="Metrics file not found. Run training pipeline first.",
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    return MetricsResponse(metrics=metrics)


@app.get("/models", tags=["monitoring"])
async def list_models():
    return {
        "ensemble_method": predictor.ensemble.method if predictor.ensemble else None,
        "ensemble_weights": predictor.ensemble.get_weights() if predictor.ensemble else {},
        "models": ["arima", "prophet", "lightgbm", "xgboost"],
        "loaded": predictor.is_loaded,
    }
