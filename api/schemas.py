from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    start_datetime: str = Field(
        ...,
        description="ISO-8601 start datetime for forecasting",
        json_schema_extra={"example": "2018-06-01T00:00:00"},
    )
    steps: int = Field(
        default=24, ge=1, le=720, description="Steps to forecast ahead (1-720)"
    )
    include_components: bool = Field(
        default=False, description="Include per-model predictions in response"
    )
    history: Optional[List[float]] = Field(
        default=None,
        min_length=1,
        description=(
            "Recent OT values in chronological order (standardised scale), "
            "ending at start_datetime - 1h. Providing at least 168 values "
            "enables accurate predictions via real lag features. "
            "When omitted, the API uses a constant-seed synthetic feature matrix."
        ),
    )


class ForecastPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    datetime: str
    prediction: float
    ridge: Optional[float] = None
    lgbm: Optional[float] = None
    xgboost: Optional[float] = None
    catboost: Optional[float] = None
    extra_trees: Optional[float] = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    steps: int
    forecast: List[ForecastPoint]
    hybrid_components: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    metrics: Dict[str, float]
