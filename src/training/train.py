import argparse
import json
import os
import pickle
import sys
import warnings

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.evaluation.metrics import compute_all_metrics
from src.models.catboost_model import CatBoostForecaster
from src.models.ensemble import EnsembleForecaster
from src.models.extra_trees_model import ExtraTreesForecaster
from src.models.lgbm_model import LGBMForecaster
from src.models.linear_model import RidgeForecaster
from src.models.xgboost_model import XGBForecaster


def load_data(cfg: dict) -> tuple:
    target_col = cfg["base"]["target_col"]
    datetime_col = cfg["base"]["datetime_col"]

    train_feat = pd.read_csv("data/processed/train_features.csv")
    val_feat = pd.read_csv("data/processed/val_features.csv")
    test_feat = pd.read_csv("data/processed/test_features.csv")

    return train_feat, val_feat, test_feat, target_col, datetime_col


def train_ridge(
    cfg: dict,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> RidgeForecaster:
    print("[train] Training Ridge...")
    model = RidgeForecaster(alpha=cfg["ridge"]["alpha"])
    model.fit(train_feat, val_feat, target_col, datetime_col)
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
    )
    model.fit(train_feat, val_feat, target_col, datetime_col)
    return model


def train_catboost(
    cfg: dict,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> CatBoostForecaster:
    print("[train] Training CatBoost...")
    c_cfg = cfg["catboost"]
    model = CatBoostForecaster(
        n_estimators=c_cfg["n_estimators"],
        learning_rate=c_cfg["learning_rate"],
        max_depth=c_cfg["max_depth"],
        subsample=c_cfg["subsample"],
        reg_lambda=c_cfg["reg_lambda"],
        min_child_samples=c_cfg["min_child_samples"],
    )
    model.fit(train_feat, val_feat, target_col, datetime_col)
    return model


def train_extra_trees(
    cfg: dict,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> ExtraTreesForecaster:
    print("[train] Training Extra Trees...")
    et_cfg = cfg["extra_trees"]
    model = ExtraTreesForecaster(
        n_estimators=et_cfg["n_estimators"],
        max_depth=et_cfg["max_depth"],   # None allowed via yaml null
        min_samples_leaf=et_cfg["min_samples_leaf"],
        max_features=et_cfg["max_features"],
    )
    model.fit(train_feat, val_feat, target_col, datetime_col)
    return model


def get_predictions(
    models: dict,
    val_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    target_col: str,
    datetime_col: str,
) -> tuple[dict, dict]:
    val_preds = {}
    test_preds = {}
    for name, model in models.items():
        val_preds[name] = model.predict(val_feat, target_col, datetime_col)
        test_preds[name] = model.predict(test_feat, target_col, datetime_col)
    return val_preds, test_preds


def log_model_metrics(model_name: str, metrics: dict, params: dict = None) -> None:
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

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    train_feat, val_feat, test_feat, target_col, datetime_col = load_data(cfg)

    y_val = val_feat[target_col].values
    y_test = test_feat[target_col].values

    # Models included in the final ensemble (from params.yaml)
    ensemble_model_names: list[str] = cfg["ensemble"]["models"]

    with mlflow.start_run(run_name="ensemble_training") as run:
        mlflow.log_params(
            {
                "horizon": cfg["data"]["horizon"],
                "test_size": cfg["base"]["test_size"],
                "val_size": cfg["base"]["val_size"],
                "ensemble_method": cfg["ensemble"]["method"],
                "ensemble_models": ",".join(ensemble_model_names),
            }
        )

        # ── Train all candidate models ──────────────────────────────────────
        with mlflow.start_run(run_name="ridge", nested=True):
            ridge = train_ridge(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(ridge.get_params())

        with mlflow.start_run(run_name="lgbm", nested=True):
            lgbm = train_lgbm(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(lgbm.get_params())

        with mlflow.start_run(run_name="xgboost", nested=True):
            xgb_model = train_xgboost(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(xgb_model.get_params())

        with mlflow.start_run(run_name="catboost", nested=True):
            catboost_model = train_catboost(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(catboost_model.get_params())

        with mlflow.start_run(run_name="extra_trees", nested=True):
            et_model = train_extra_trees(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(et_model.get_params())

        # ── All trained models (for observability logging) ──────────────────
        all_models = {
            "ridge": ridge,
            "lgbm": lgbm,
            "xgboost": xgb_model,
            "catboost": catboost_model,
            "extra_trees": et_model,
        }

        print("\n[train] Generating predictions for all models...")
        val_preds_all, test_preds_all = get_predictions(
            all_models, val_feat, test_feat, target_col, datetime_col
        )

        # Log metrics for every model to MLflow (full observability)
        all_metrics = {}
        for name, preds in test_preds_all.items():
            m = compute_all_metrics(y_test, preds)
            all_metrics[name] = m
            log_model_metrics(name, m)
            print(
                f"[eval] {name:12s} | RMSE={m['rmse']:.4f} "
                f"| MAE={m['mae']:.4f} | SMAPE={m['smape']:.2f}% | R²={m['r2']:.4f}"
            )

        # ── Fit ensemble on the 3 selected models only ──────────────────────
        print(f"\n[train] Fitting ensemble on: {ensemble_model_names}")
        val_preds_ens = {k: val_preds_all[k] for k in ensemble_model_names}
        test_preds_ens = {k: test_preds_all[k] for k in ensemble_model_names}

        ensemble = EnsembleForecaster(
            method=cfg["ensemble"]["method"],
            min_weight=cfg["ensemble"].get("min_weight", 0.10),
        )
        ensemble.fit_weights(val_preds_ens, y_val)
        ensemble_preds = ensemble.predict(test_preds_ens)

        ensemble_metrics = compute_all_metrics(y_test, ensemble_preds)
        all_metrics["ensemble"] = ensemble_metrics
        log_model_metrics("ensemble", ensemble_metrics)
        mlflow.log_metrics(
            {f"ensemble_weight_{k}": v for k, v in ensemble.get_weights().items()}
        )
        print(
            f"[eval] {'ensemble':12s} | RMSE={ensemble_metrics['rmse']:.4f} "
            f"| MAE={ensemble_metrics['mae']:.4f} | SMAPE={ensemble_metrics['smape']:.2f}% "
            f"| R²={ensemble_metrics['r2']:.4f}"
        )

        # ── Persist all models ───────────────────────────────────────────────
        for name, obj in [*all_models.items(), ("ensemble", ensemble)]:
            with open(f"models/{name}.pkl", "wb") as f:
                pickle.dump(obj, f)

        mlflow.log_artifacts("models", artifact_path="models")
        mlflow.log_artifact(config_path)

        flat_metrics = {
            f"{model_name}_{k}": round(float(v), 6)
            for model_name, mets in all_metrics.items()
            for k, v in mets.items()
        }
        with open("metrics/metrics.json", "w") as f:
            json.dump(flat_metrics, f, indent=2)

        pred_df = pd.DataFrame(
            {
                "ds": (
                    test_feat[datetime_col].values
                    if datetime_col in test_feat.columns
                    else range(len(y_test))
                ),
                "y_true": y_test,
                "y_pred": ensemble_preds,
                **{f"y_{k}": v for k, v in test_preds_all.items()},
            }
        )
        pred_df.to_csv("metrics/predictions.csv", index=False)
        mlflow.log_artifact("metrics/predictions.csv")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/models"
        client = MlflowClient()
        model_name = cfg["mlflow"]["experiment_name"] + "-ensemble"
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass
        client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
        print(f"\n[train] ✓ Run ID: {run_id}")
        print(f"[train] ✓ Ensemble members: {ensemble_model_names}")
        print(f"[train] ✓ Ensemble weights: {ensemble.get_weights()}")
        print(f"[train] ✓ Model registered as: {model_name}")
        print("[train] ✓ Metrics saved → metrics/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
