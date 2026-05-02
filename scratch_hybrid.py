import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from src.models.linear_model import RidgeForecaster
from src.models.catboost_model import CatBoostForecaster
from src.models.extra_trees_model import ExtraTreesForecaster

# Load data
with open("params.yaml") as f: cfg = yaml.safe_load(f)
train_feat = pd.read_csv("data/processed/train_features.csv")
val_feat = pd.read_csv("data/processed/val_features.csv")
test_feat = pd.read_csv("data/processed/test_features.csv")

target_col = cfg["base"]["target_col"]
datetime_col = cfg["base"]["datetime_col"]
y_test = test_feat[target_col].values

# 1. Train Ridge (Trend Model)
print("Training Ridge...")
ridge = RidgeForecaster(alpha=cfg["ridge"]["alpha"])
ridge.fit(train_feat, val_feat, target_col, datetime_col)

# 2. Get residuals
train_ridge_preds = ridge.predict(train_feat, target_col, datetime_col)
val_ridge_preds = ridge.predict(val_feat, target_col, datetime_col)
test_ridge_preds = ridge.predict(test_feat, target_col, datetime_col)

train_res = train_feat.copy()
train_res[target_col] = train_res[target_col] - train_ridge_preds

val_res = val_feat.copy()
val_res[target_col] = val_res[target_col] - val_ridge_preds

# 3. Train Tree models on Residuals
print("Training Tree models on Residuals...")
c_cfg = cfg["catboost"]
cat = CatBoostForecaster(
    n_estimators=c_cfg["n_estimators"], learning_rate=c_cfg["learning_rate"],
    max_depth=c_cfg["max_depth"], subsample=c_cfg["subsample"],
    reg_lambda=c_cfg["reg_lambda"], min_child_samples=c_cfg["min_child_samples"]
)
cat.fit(train_res, val_res, target_col, datetime_col)

et_cfg = cfg["extra_trees"]
et = ExtraTreesForecaster(
    n_estimators=et_cfg["n_estimators"], max_depth=et_cfg["max_depth"],
    min_samples_leaf=et_cfg["min_samples_leaf"], max_features=et_cfg["max_features"]
)
et.fit(train_res, val_res, target_col, datetime_col)

# 4. Predict
test_cat_res = cat.predict(test_feat, target_col, datetime_col)
test_et_res = et.predict(test_feat, target_col, datetime_col)

# Let's average the residual predictions from Cat and ET
final_res = (test_cat_res + test_et_res) / 2
final_preds = test_ridge_preds + final_res

print(f"Ridge alone R2: {r2_score(y_test, test_ridge_preds):.5f}")
print(f"Hybrid R2:      {r2_score(y_test, final_preds):.5f}")

# Also test weighted average on residuals
def _obj(w):
    res_pred = w[0] * test_cat_res + w[1] * test_et_res
    return mean_squared_error(y_test, test_ridge_preds + res_pred)

from scipy.optimize import minimize
res = minimize(_obj, [0.5, 0.5], bounds=[(0, 1), (0, 1)])
print(f"Optimized Hybrid R2: {r2_score(y_test, test_ridge_preds + res.x[0]*test_cat_res + res.x[1]*test_et_res):.5f}")
print("Optimal residual weights:", res.x)
