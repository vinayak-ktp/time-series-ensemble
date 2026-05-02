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
    print("[train] Training Ridge (Base Trend Model)...")
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

    # Models included in the final hybrid ensemble
    base_model_name = cfg["hybrid"]["base_model"]
    residual_model_names = cfg["hybrid"]["residual_models"]

    with mlflow.start_run(run_name="hybrid_training") as run:
        mlflow.log_params(
            {
                "horizon": cfg["data"]["horizon"],
                "test_size": cfg["base"]["test_size"],
                "val_size": cfg["base"]["val_size"],
                "hybrid_base_model": base_model_name,
                "hybrid_residual_models": ",".join(residual_model_names),
            }
        )

        all_models = {}

        # ── 1. Train Base Model (Ridge) ─────────────────────────────────────
        with mlflow.start_run(run_name="ridge", nested=True):
            ridge = train_ridge(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(ridge.get_params())
            all_models["ridge"] = ridge

        # ── 2. Calculate Residuals ──────────────────────────────────────────
        print("\n[train] Calculating residuals from Ridge...")
        train_ridge_preds = ridge.predict(train_feat, target_col, datetime_col)
        val_ridge_preds = ridge.predict(val_feat, target_col, datetime_col)
        test_ridge_preds = ridge.predict(test_feat, target_col, datetime_col)

        train_res_feat = train_feat.copy()
        train_res_feat[target_col] = train_feat[target_col] - train_ridge_preds

        val_res_feat = val_feat.copy()
        val_res_feat[target_col] = val_feat[target_col] - val_ridge_preds

        # ── 3. Train Residual Models (CatBoost, ExtraTrees) ─────────────────
        print(f"[train] Training residual models on errors: {residual_model_names}")
        with mlflow.start_run(run_name="catboost", nested=True):
            catboost_model = train_catboost(cfg, train_res_feat, val_res_feat, target_col, datetime_col)
            mlflow.log_params(catboost_model.get_params())
            all_models["catboost"] = catboost_model

        with mlflow.start_run(run_name="extra_trees", nested=True):
            et_model = train_extra_trees(cfg, train_res_feat, val_res_feat, target_col, datetime_col)
            mlflow.log_params(et_model.get_params())
            all_models["extra_trees"] = et_model

        # Also train XGBoost/LGBM on normal target for observability, not part of hybrid
        with mlflow.start_run(run_name="lgbm", nested=True):
            lgbm = train_lgbm(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(lgbm.get_params())
            all_models["lgbm"] = lgbm

        with mlflow.start_run(run_name="xgboost", nested=True):
            xgb_model = train_xgboost(cfg, train_feat, val_feat, target_col, datetime_col)
            mlflow.log_params(xgb_model.get_params())
            all_models["xgboost"] = xgb_model

        # ── 4. Generate Predictions & Evaluate ──────────────────────────────
        test_preds_all = {}
        all_metrics = {}

        # Evaluate models trained on normal target (Ridge, LGBM, XGBoost)
        for name in ["ridge", "lgbm", "xgboost"]:
            test_preds_all[name] = all_models[name].predict(test_feat, target_col, datetime_col)
            m = compute_all_metrics(y_test, test_preds_all[name])
            all_metrics[name] = m
            log_model_metrics(name, m)
            print(
                f"[eval] {name:12s} | RMSE={m['rmse']:.4f} "
                f"| MAE={m['mae']:.4f} | SMAPE={m['smape']:.2f}% | R²={m['r2']:.4f}"
            )

        # Generate residual predictions from CatBoost and ExtraTrees
        test_cat_res = all_models["catboost"].predict(test_feat, target_col, datetime_col)
        test_et_res = all_models["extra_trees"].predict(test_feat, target_col, datetime_col)
        test_preds_all["catboost_residuals"] = test_cat_res
        test_preds_all["extra_trees_residuals"] = test_et_res

        # ── 5. Hybrid Prediction (Ridge + Average of Residuals) ─────────────
        hybrid_res = (test_cat_res + test_et_res) / 2
        hybrid_preds = test_ridge_preds + hybrid_res

        hybrid_metrics = compute_all_metrics(y_test, hybrid_preds)
        all_metrics["hybrid"] = hybrid_metrics
        log_model_metrics("hybrid", hybrid_metrics)
        print(
            f"[eval] {'hybrid':12s} | RMSE={hybrid_metrics['rmse']:.4f} "
            f"| MAE={hybrid_metrics['mae']:.4f} | SMAPE={hybrid_metrics['smape']:.2f}% "
            f"| R²={hybrid_metrics['r2']:.4f}"
        )

        # ── Persist all models ───────────────────────────────────────────────
        # Note: We only persist the models needed for the hybrid, or all of them if preferred.
        # Let's save all for consistency.
        for name, obj in all_models.items():
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
                "y_hybrid": hybrid_preds,
                "y_ridge": test_preds_all["ridge"],
                "y_catboost_res": test_cat_res,
                "y_extra_trees_res": test_et_res,
            }
        )
        pred_df.to_csv("metrics/predictions.csv", index=False)
        mlflow.log_artifact("metrics/predictions.csv")

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/models"
        client = MlflowClient()
        model_name = cfg["mlflow"]["experiment_name"] + "-hybrid"
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass
        client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
        
        print(f"\n[train] ✓ Run ID: {run_id}")
        print(f"[train] ✓ Hybrid Base: {base_model_name}")
        print(f"[train] ✓ Hybrid Residuals: {residual_model_names}")
        print(f"[train] ✓ Model registered as: {model_name}")
        print("[train] ✓ Metrics saved → metrics/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(args.config)
