# MLOps Time Series Ensemble Pipeline

[![CI/CD](https://github.com/vinayak-ktp/time-series-ensemble/actions/workflows/ci.yml/badge.svg)](https://github.com/vinayak-ktp/time-series-ensemble/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-pipeline-945DD6.svg)](https://dvc.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ED.svg)](https://docker.com)

A **production-grade MLOps project** demonstrating end-to-end best practices for machine learning engineering. Trains an ensemble of time series forecasting models on the **ETTh1 (Electricity Transformer Temperature)** dataset and serves predictions via a REST API.

---

## Architecture

```
┌─────────────┐    ┌─────────────────────────────────────────────────────┐
│   ETTh1     │───▶│               DVC Pipeline                          │
│  Dataset    │    │  ingest → preprocess → featurize → train            │
└─────────────┘    └────────┬─────────────────────────────┬──────────────┘
                            │                             │
                     ┌──────▼──────┐              ┌───────▼───────┐
                     │   MLflow    │              │ Model Pickles │
                     │  Tracking   │              │ (5 artifacts) │
                     └─────────────┘              └───────┬───────┘
                                                          │
                                                  ┌───────▼────────┐
                                                  │  FastAPI + UV  │
                                                  │  /predict API  │
                                                  └────────────────┘
```

### Hybrid Architecture

| Model | Type | Role | Strength |
|-------|------|------|----------|
| Ridge | Linear Regression | Base Trend | Fast, stable baseline |
| LightGBM | Gradient boosting | Observability | Fast, feature-based |
| XGBoost | Gradient boosting | Observability | Robust, feature-based |
| CatBoost | Gradient boosting | Residuals | Finer residual fitting |
| ExtraTrees | Tree Ensemble | Residuals | Stable variance reduction |
| **Hybrid** | **Base + Residuals** | **Ensemble** | **Best of all worlds** |

---

## Evaluation Results

Latest test set results (`metrics/metrics.json`):

| Model | MAE | RMSE | MAPE (%) | SMAPE (%) | R² |
|-------|-----|------|----------|-----------|----|
| **Hybrid (Ridge + Residuals)** | **0.0519** | **0.0757** | 10.74 | **7.80** | **0.9645** |
| Ridge | 0.0526 | 0.0765 | **10.55** | 7.87 | 0.9638 |
| LightGBM | 0.0581 | 0.0816 | 11.19 | 8.44 | 0.9588 |
| XGBoost | 0.0594 | 0.0837 | 11.30 | 8.58 | 0.9567 |

---

## Project Structure

```
mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD: lint → test → DVC → Docker
├── .dvc/                       # DVC internal files
├── api/
│   ├── main.py                 # FastAPI app (4 endpoints)
│   ├── predictor.py            # Model loading & inference
│   └── schemas.py              # Pydantic request/response schemas
├── data/
│   ├── raw/                    # ETTh1.csv (DVC tracked)
│   └── processed/              # Splits + features (DVC tracked)
├── docker/
│   ├── Dockerfile.train        # Training container
│   └── Dockerfile.api          # API container
├── metrics/
│   ├── metrics.json            # DVC metrics (all models + ensemble)
│   └── predictions.csv         # DVC plots data
├── models/                     # Saved model pickles
├── src/
│   ├── data/
│   │   ├── ingestion.py        # Downloads ETTh1 dataset
│   │   └── preprocessing.py    # Chronological split + scaling
│   ├── features/
│   │   └── engineering.py      # Lag, rolling, cyclical features
│   ├── models/
│   │   ├── linear.py           # Ridge baseline
│   │   ├── lgbm.py
│   │   ├── xgboost.py
│   │   ├── catboost.py
│   │   └── extra_trees.py      # Tree ensemble
│   ├── training/
│   │   └── train.py            # Orchestrator with MLflow tracking
│   └── evaluation/
│       └── metrics.py          # MAE, RMSE, MAPE, SMAPE, R²
├── tests/
│   ├── test_preprocessing.py   # 15 unit tests
│   ├── test_ensemble.py        # 8 ensemble tests
│   └── test_api.py             # 8 API integration tests
├── .gitignore
├── .pre-commit-config.yaml     # black, isort, flake8 hooks
├── docker-compose.yml          # 3 services: train, API, MLflow
├── dvc.yaml                    # 4-stage pipeline
├── params.yaml                 # All hyperparameters
└── requirements.txt
```

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/vinayak-ktp/time-series-ensemble.git
cd time-series-ensemble

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize DVC & Git

```bash
git init
git add .
git commit -m "feat: initial MLOps project setup"

dvc init
dvc add data/raw/ETTh1.csv   # after running pipeline
git add .dvc .gitignore
git commit -m "feat: initialize DVC tracking"
```

### 3. Run the DVC Pipeline

```bash
# Run all 4 stages: ingest → preprocess → featurize → train
dvc repro

# View metrics
dvc metrics show

# View pipeline DAG
dvc dag
```

### 4. Launch MLflow UI

```bash
mlflow ui --backend-store-uri mlruns --port 5000
# Open http://localhost:5000
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root info |
| `GET` | `/health` | Health check + model status |
| `POST` | `/predict` | **Ensemble forecast** |
| `GET` | `/metrics` | Latest test-set metrics |
| `GET` | `/models` | Loaded models + weights |

### Example: Forecast 24 hours

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "start_datetime": "2018-06-01T00:00:00",
    "steps": 24,
    "include_components": true
  }'
```

Response:
```json
{
  "model": "mlops-timeseries-ensemble",
  "steps": 24,
  "forecast": [
    {
      "datetime": "2018-06-01T00:00:00",
      "prediction": 3.142,
      "ridge": 3.01,
      "catboost": 0.08,
      "extra_trees": 0.052
    }
  ],
  "hybrid_components": {
    "base": "ridge",
    "residuals": ["catboost", "extra_trees"]
  }
}
```

---

## Docker

### Run with Docker Compose

```bash
# Start MLflow server + API
docker-compose up api mlflow

# Run training (one-shot)
docker-compose --profile train run train

# Full stack
docker-compose up
```

### Build individual images

```bash
# API image
docker build -f docker/Dockerfile.api -t mlops-api:latest .

# Training image
docker build -f docker/Dockerfile.train -t mlops-train:latest .
```

---

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocessing.py -v
pytest tests/test_api.py -v
pytest tests/test_ensemble.py -v
```

---

## Configuration

All hyperparameters live in `params.yaml`. To run an experiment with different settings:

```bash
# Edit params.yaml then re-run
dvc repro

# Compare experiments
dvc metrics diff
dvc params diff
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base.horizon` | 24 | Forecast horizon (hours) |
| `features.lag_periods` | [1,2,3,6,12,24,48,168] | Lag feature periods |
| `hybrid.base_model` | `ridge` | Base model for the hybrid architecture |
| `hybrid.residual_models` | `[catboost, extra_trees]` | Models trained on residuals |
| `lightgbm.n_estimators` | 500 | Max trees (early stopping) |

---

## CI/CD Pipeline

GitHub Actions runs on every push/PR:

```
lint (flake8, black, isort)
    └── test (pytest + coverage)
            └── dvc-pipeline (on main only)
            └── docker-build (build + test + push to Docker Hub)
```

Pre-commit hooks (run locally):
```bash
pre-commit install
pre-commit run --all-files
```

---

## Dataset

**ETTh1** — Electricity Transformer Temperature (Hourly)
- **Source**: [Zhou et al., 2021](https://github.com/zhouhaoyi/ETDataset)
- **Size**: 17,420 rows × 7 features
- **Target**: `OT` (Oil Temperature)
- **Frequency**: Hourly
- **Date range**: 2016-07 to 2018-06

---

## MLflow Experiment Tracking

Each `dvc repro` creates a nested MLflow run:

```
experiment: mlops-timeseries-ensemble
  └── run: hybrid_training
      ├── nested: ridge       (params + metrics)
      ├── nested: catboost    (params + metrics)
      ├── nested: extra_trees (params + metrics)
      ├── nested: lgbm        (params + metrics)
      ├── nested: xgboost     (params + metrics)
      └── hybrid metrics + artifacts
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn, statsmodels |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-cov |
| Code Quality | black, isort, flake8, pre-commit |
| Config | YAML (params.yaml, dvc.yaml) |
