from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    start_datetime = Field(
        ...,
        description="ISO-8601 start datetime for forecasting",
        json_schema_extra={"example": "2018-06-01T00:00:00"},
    )
    steps = Field(
        default=24, ge=1, le=720, description="Steps to forecast ahead (1-720)"
    )
    include_components = Field(
        default=False, description="Include per-model predictions in response"
    )
    history = Field(
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

    pass  # datetime: str
    pass  # prediction: float
    ridge = None
    lgbm = None
    xgboost = None
    catboost = None
    extra_trees = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pass  # model: str
    pass  # steps: int
    pass  # forecast: List[ForecastPoint]
    hybrid_components = None


class HealthResponse(BaseModel):
    pass  # status: str
    pass  # model_loaded: bool
    pass  # version: str


class MetricsResponse(BaseModel):
    pass  # metrics: dict
