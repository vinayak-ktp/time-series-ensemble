# MLOps Time Series Ensemble Pipeline

[![CI/CD](https://github.com/yourusername/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/mlops-pipeline/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-pipeline-945DD6.svg)](https://dvc.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ED.svg)](https://docker.com)

A **production-grade MLOps project** demonstrating end-to-end best practices for machine learning engineering. Trains an ensemble of time series forecasting models on the **ETTh1 (Electricity Transformer Temperature)** dataset and serves predictions via a REST API.

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────────────────────────────────────────────┐
│   ETTh1     │───▶│               DVC Pipeline                           │
│  Dataset    │    │  ingest → preprocess → featurize → train             │
└─────────────┘    └────────┬──────────────────────────────┬──────────────┘
                            │                              │
                     ┌──────▼──────┐              ┌───────▼────────┐
                     │   MLflow    │              │  Model Pickles  │
                     │  Tracking   │              │  (5 artifacts)  │
                     └─────────────┘              └───────┬────────┘
                                                          │
                                                  ┌───────▼────────┐
                                                  │  FastAPI + UV  │
                                                  │  /predict API  │
                                                  └────────────────┘
```

### Ensemble Architecture

| Model | Type | Library | Strength |
|-------|------|---------|----------|
| ARIMA(2,1,2) | Statistical | statsmodels | Trend + autocorrelation |
| Prophet | Additive decomposition | prophet | Multi-seasonality + holidays |
| LightGBM | Gradient boosting | lightgbm | Feature-based, fast |
| XGBoost | Gradient boosting | xgboost | Feature-based, robust |
| **Ensemble** | **SLSQP weighted avg** | scipy | **Best of all worlds** |

---

## 📁 Project Structure

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
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── lgbm_model.py
│   │   ├── xgboost_model.py
│   │   └── ensemble.py         # Weighted avg + Ridge stacking
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

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone <your-repo-url>
cd mlops-pipeline

# Activate virtual environment
source venv/bin/activate

# (Already installed) Verify dependencies
pip list | grep -E "mlflow|dvc|fastapi|lightgbm"
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

## 📡 API Endpoints

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
      "arima": 3.01,
      "prophet": 3.22,
      "lgbm": 3.19,
      "xgboost": 3.15
    }
  ],
  "ensemble_weights": {"arima": 0.18, "prophet": 0.30, "lgbm": 0.26, "xgboost": 0.26}
}
```

---

## 🐳 Docker

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

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocessing.py -v
pytest tests/test_api.py -v
pytest tests/test_ensemble.py -v
```

---

## ⚙️ Configuration

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
| `ensemble.method` | `weighted_average` | `simple_average` / `weighted_average` / `stacking` |
| `arima.p/d/q` | 2/1/2 | ARIMA order |
| `lightgbm.n_estimators` | 500 | Max trees (early stopping) |

---

## 🔄 CI/CD Pipeline

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

## 📊 Dataset

**ETTh1** — Electricity Transformer Temperature (Hourly)
- **Source**: [Zhou et al., 2021](https://github.com/zhouhaoyi/ETDataset)
- **Size**: 17,420 rows × 7 features
- **Target**: `OT` (Oil Temperature)
- **Frequency**: Hourly
- **Date range**: 2016-07 to 2018-06

---

## 📈 MLflow Experiment Tracking

Each `dvc repro` creates a nested MLflow run:

```
📁 experiment: mlops-timeseries-ensemble
   └── 🏃 run: ensemble_training
       ├── 🏃 nested: arima       (params + metrics)
       ├── 🏃 nested: prophet     (params + metrics)
       ├── 🏃 nested: lgbm        (params + metrics)
       ├── 🏃 nested: xgboost     (params + metrics)
       └── 📊 ensemble metrics + weights + artifacts
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn, statsmodels, prophet, lightgbm, xgboost |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-cov |
| Code Quality | black, isort, flake8, pre-commit |
| Config | YAML (params.yaml, dvc.yaml) |
