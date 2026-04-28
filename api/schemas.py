from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    start_datetime: str = Field(
        ...,
        description="ISO-8601 start datetime for forecasting",
        json_schema_extra={"example": "2018-06-01T00:00:00"},
    )
    steps: int = Field(default=24, ge=1, le=720, description="Steps to forecast ahead (1-720)")
    include_components: bool = Field(
        default=False, description="Include per-model predictions in response"
    )


class ForecastPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    datetime: str
    prediction: float
    arima: Optional[float] = None
    prophet: Optional[float] = None
    lgbm: Optional[float] = None
    xgboost: Optional[float] = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    steps: int
    forecast: List[ForecastPoint]
    ensemble_weights: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    metrics: dict
