"""
Pydantic schemas for the MLOps forecasting API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PredictRequest(BaseModel):
    """Request for multi-step forecast from a given start timestamp."""
    start_datetime: str = Field(
        ...,
        description="ISO-8601 start datetime for forecasting (e.g. '2018-06-01T00:00:00')",
        json_schema_extra={"example": "2018-06-01T00:00:00"},
    )
    steps: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Number of steps to forecast ahead (1–720 hours)",
    )
    include_components: bool = Field(
        default=False,
        description="Whether to include individual model predictions in response",
    )


class ForecastPoint(BaseModel):
    datetime: str
    prediction: float
    arima: Optional[float] = None
    prophet: Optional[float] = None
    lgbm: Optional[float] = None
    xgboost: Optional[float] = None


class PredictResponse(BaseModel):
    model: str
    steps: int
    forecast: List[ForecastPoint]
    ensemble_weights: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """Last-run test metrics."""
    metrics: dict
