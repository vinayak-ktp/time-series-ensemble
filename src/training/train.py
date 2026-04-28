"""
Main Training Script
Orchestrates the full training pipeline with MLflow tracking:
  1. Load processed + featurized data
  2. Train ARIMA, Prophet, LightGBM, XGBoost models
  3. Evaluate each model individually on test set
  4. Train ensemble and evaluate
  5. Log all params, metrics, and artifacts to MLflow
  6. Register best model to MLflow Model Registry
"""
import argparse
import json
import os
import pickle
import sys
import warnings
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.lgbm_model import LGBMForecaster
from src.models.xgboost_model import XGBForecaster
from src.models.ensemble import EnsembleForecaster
from src.evaluation.metrics import compute_all_metrics


def load_data(cfg: dict) -> tuple:
    target_col = cfg["base"]["target_col"]
    datetime_col = cfg["base"]["datetime_col"]

    train_raw = pd.read_csv(cfg["data"]["processed_train_path"])
    val_raw = pd.read_csv(cfg["data"]["processed_val_path"])
    test_raw = pd.read_csv(cfg["data"]["processed_test_path"])

    train_feat = pd.read_csv("data/processed/train_features.csv")
    val_feat = pd.read_csv("data/processed/val_features.csv")
    test_feat = pd.read_csv("data/processed/test_features.csv")

    return (
        train_raw, val_raw, test_raw,
        train_feat, val_feat, test_feat,
        target_col, datetime_col,
    )


def train_arima(cfg: dict, train_raw: pd.DataFrame, target_col: str) -> ARIMAForecaster:
    print("\n[train] Training ARIMA...")
    arima_cfg = cfg["arima"]
    model = ARIMAForecaster(
        p=arima_cfg["p"],
        d=arima_cfg["d"],
        q=arima_cfg["q"],
        horizon=cfg["data"]["horizon"],
    )
    model.fit(train_raw[target_col])
    return model


def train_prophet(
    cfg: dict,
    train_raw: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> ProphetForecaster:
    print("[train] Training Prophet...")
    p_cfg = cfg["prophet"]
    model = ProphetForecaster(
        changepoint_prior_scale=p_cfg["changepoint_prior_scale"],
        seasonality_prior_scale=p_cfg["seasonality_prior_scale"],
        seasonality_mode=p_cfg["seasonality_mode"],
        yearly_seasonality=p_cfg["yearly_seasonality"],
        weekly_seasonality=p_cfg["weekly_seasonality"],
        daily_seasonality=p_cfg["daily_seasonality"],
        horizon=cfg["data"]["horizon"],
        freq=cfg["data"]["freq"],
    )
    model.fit(train_raw, datetime_col, target_col)
    return model


def train_lgbm(
    cfg: dict,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> LGBMForecaster:
    print("[train] Training LightGBM...")
    l_cfg = cfg["lightgbm"]
    model = LGBMForecaster(
        n_estimators=l_cfg["n_estimators"],
        learning_rate=l_cfg["learning_rate"],
        max_depth=l_cfg["max_depth"],
        num_leaves=l_cfg["num_leaves"],
        min_child_samples=l_cfg["min_child_samples"],
        subsample=l_cfg["subsample"],
        colsample_bytree=l_cfg["colsample_bytree"],
        reg_alpha=l_cfg["reg_alpha"],
        reg_lambda=l_cfg["reg_lambda"],
        random_state=cfg["base"]["random_seed"],
    )
    model.fit(train_feat, val_feat, target_col, datetime_col)
    return model


def train_xgboost(
    cfg: dict,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> XGBForecaster:
    print("[train] Training XGBoost...")
    x_cfg = cfg["xgboost"]
    model = XGBForecaster(
        n_estimators=x_cfg["n_estimators"],
        learning_rate=x_cfg["learning_rate"],
        max_depth=x_cfg["max_depth"],
        subsample=x_cfg["subsample"],
        colsample_bytree=x_cfg["colsample_bytree"],
        reg_alpha=x_cfg["reg_alpha"],
        reg_lambda=x_cfg["reg_lambda"],
        min_child_weight=x_cfg["min_child_weight"],
        random_state=cfg["base"]["random_seed"],
    )
    model.fit(train_feat, val_feat, target_col, datetime_col)
    return model


def get_predictions(
    models: dict,
    val_raw: pd.DataFrame,
    val_feat: pd.DataFrame,
    test_raw: pd.DataFrame,
    test_feat: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> tuple[dict, dict]:
    """Get val and test predictions from all base models."""
    arima: ARIMAForecaster = models["arima"]
    prophet: ProphetForecaster = models["prophet"]
    lgbm: LGBMForecaster = models["lgbm"]
    xgb: XGBForecaster = models["xgboost"]

    val_preds = {
        "arima": arima.predict_in_sample(
            len(val_raw) - len(val_raw), len(val_raw) - 1
        ) if False else np.full(len(val_feat), np.nan),  # placeholder
        "prophet": prophet.predict_on_df(val_feat, datetime_col),
        "lgbm": lgbm.predict(val_feat, target_col, datetime_col),
        "xgboost": xgb.predict(val_feat, target_col, datetime_col),
    }
    # ARIMA val predictions using in-sample forecast approach
    # Re-forecast from position len(train) for val length
    arima_val_preds = lgbm.predict(val_feat, target_col, datetime_col)  # fallback to lgbm shape
    arima_val_preds = prophet.predict_on_df(val_feat, datetime_col)
    val_preds["arima"] = arima_val_preds  # prophet-like for ARIMA on val (same series)

    test_preds = {
        "arima": prophet.predict_on_df(test_feat, datetime_col),  # same shape
        "prophet": prophet.predict_on_df(test_feat, datetime_col),
        "lgbm": lgbm.predict(test_feat, target_col, datetime_col),
        "xgboost": xgb.predict(test_feat, target_col, datetime_col),
    }
    # Re-run ARIMA properly
    arima_test_preds = arima.predict(steps=len(test_feat))
    if len(arima_test_preds) < len(test_feat):
        arima_test_preds = np.pad(
            arima_test_preds,
            (0, len(test_feat) - len(arima_test_preds)),
            constant_values=arima_test_preds[-1],
        )
    arima_val_preds_final = arima.predict(steps=len(val_feat))
    if len(arima_val_preds_final) < len(val_feat):
        arima_val_preds_final = np.pad(
            arima_val_preds_final,
            (0, len(val_feat) - len(arima_val_preds_final)),
            constant_values=arima_val_preds_final[-1],
        )
    val_preds["arima"] = arima_val_preds_final
    test_preds["arima"] = arima_test_preds[: len(test_feat)]

    return val_preds, test_preds


def log_model_metrics(
    run: mlflow.ActiveRun,
    model_name: str,
    metrics: dict,
    params: dict = None,
) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(f"{model_name}_{k}", v)
    if params:
        for k, v in params.items():
            mlflow.log_param(f"{model_name}_{k}", v)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("metrics", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Setup MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    (
        train_raw, val_raw, test_raw,
        train_feat, val_feat, test_feat,
        target_col, datetime_col,
    ) = load_data(cfg)

    y_val = val_feat[target_col].values
    y_test = test_feat[target_col].values

    with mlflow.start_run(run_name="ensemble_training") as run:
        # ── Log all params from params.yaml ──────────────────────────────
        mlflow.log_params({
            "horizon": cfg["data"]["horizon"],
            "random_seed": cfg["base"]["random_seed"],
            "test_size": cfg["base"]["test_size"],
            "val_size": cfg["base"]["val_size"],
        })
        mlflow.log_param("ensemble_method", cfg["ensemble"]["method"])

        # ── Train base models ─────────────────────────────────────────────
        with mlflow.start_run(run_name="arima", nested=True):
            arima = train_arima(cfg, train_raw, target_col)
            mlflow.log_params(arima.get_params())

        with mlflow.start_run(run_name="prophet", nested=True):
            prophet = train_prophet(cfg, train_raw, target_col, datetime_col)
            mlflow.log_params(prophet.get_params())

        with mlflow.start_run(run_name="lgbm", nested=True):
            lgbm = train_lgbm(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(lgbm.get_params())

        with mlflow.start_run(run_name="xgboost", nested=True):
            xgb_model = train_xgboost(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(xgb_model.get_params())

        models = {
            "arima": arima,
            "prophet": prophet,
            "lgbm": lgbm,
            "xgboost": xgb_model,
        }

        # ── Get predictions ───────────────────────────────────────────────
        print("[train] Generating predictions...")
        val_preds, test_preds = get_predictions(
            models, val_raw, val_feat, test_raw, test_feat, target_col, datetime_col
        )

        # ── Evaluate individual models ────────────────────────────────────
        all_metrics = {}
        for name, preds in test_preds.items():
            m = compute_all_metrics(y_test, preds)
            all_metrics[name] = m
            log_model_metrics(run, name, m)
            print(f"[eval] {name:10s} | RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} | R²={m['r2']:.4f}")

        # ── Train ensemble ────────────────────────────────────────────────
        print("[train] Fitting ensemble...")
        ensemble = EnsembleForecaster(method=cfg["ensemble"]["method"])
        ensemble.fit_weights(val_preds, y_val)
        ensemble_preds = ensemble.predict(test_preds)
        ensemble_metrics = compute_all_metrics(y_test, ensemble_preds)
        all_metrics["ensemble"] = ensemble_metrics
        log_model_metrics(run, "ensemble", ensemble_metrics)
        mlflow.log_metrics({f"ensemble_weight_{k}": v for k, v in ensemble.get_weights().items()})
        print(
            f"[eval] {'ensemble':10s} | RMSE={ensemble_metrics['rmse']:.4f} "
            f"| MAE={ensemble_metrics['mae']:.4f} | R²={ensemble_metrics['r2']:.4f}"
        )

        # ── Save artifacts ────────────────────────────────────────────────
        with open("models/arima.pkl", "wb") as f:
            pickle.dump(arima, f)
        with open("models/prophet.pkl", "wb") as f:
            pickle.dump(prophet, f)
        with open("models/lgbm.pkl", "wb") as f:
            pickle.dump(lgbm, f)
        with open("models/xgboost.pkl", "wb") as f:
            pickle.dump(xgb_model, f)
        with open("models/ensemble.pkl", "wb") as f:
            pickle.dump(ensemble, f)

        mlflow.log_artifacts("models", artifact_path="models")
        mlflow.log_artifact(config_path)

        # ── Save metrics.json (DVC metrics) ──────────────────────────────
        flat_metrics = {}
        for model_name, mets in all_metrics.items():
            for k, v in mets.items():
                flat_metrics[f"{model_name}_{k}"] = round(v, 6)
        with open("metrics/metrics.json", "w") as f:
            json.dump(flat_metrics, f, indent=2)

        # ── Save predictions CSV (DVC plots) ─────────────────────────────
        pred_df = pd.DataFrame({
            "ds": test_feat[datetime_col].values if datetime_col in test_feat.columns else range(len(y_test)),
            "y_true": y_test,
            "y_pred": ensemble_preds,
            **{f"y_{k}": v for k, v in test_preds.items()},
        })
        pred_df.to_csv("metrics/predictions.csv", index=False)
        mlflow.log_artifact("metrics/predictions.csv")

        # ── Register ensemble model ───────────────────────────────────────
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/models"
        client = MlflowClient()
        model_name = cfg["mlflow"]["experiment_name"] + "-ensemble"
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass
        client.create_model_version(
            name=model_name, source=model_uri, run_id=run_id
        )
        print(f"\n[train] ✓ Run ID: {run_id}")
        print(f"[train] ✓ Model registered as: {model_name}")
        print(f"[train] ✓ Metrics saved → metrics/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
