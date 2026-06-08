# %%
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# %%
DATA_PATH = Path("data/enriched_yield_dataset.csv")
TARGET = "productivity"
RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(DATA_PATH)


# %%
sowing_date = pd.to_datetime(df["sowing_date"], errors="coerce")
harvesting_date = pd.to_datetime(df["harvesting_date"], errors="coerce")

df["sowing_month"] = sowing_date.dt.month
df["sowing_dayofyear"] = sowing_date.dt.dayofyear
df["harvesting_month"] = harvesting_date.dt.month
df["harvesting_dayofyear"] = harvesting_date.dt.dayofyear
df["season_mid_dayofyear"] = (df["sowing_dayofyear"] + df["harvesting_dayofyear"]) / 2

ordered_df = df.sort_values(["field_id", "year"]).copy()
field_productivity = ordered_df.groupby("field_id")[TARGET]

ordered_df["field_prev_productivity"] = field_productivity.shift(1)
ordered_df["field_productivity_history_count"] = field_productivity.cumcount()
ordered_df["field_productivity_history_mean"] = (
    field_productivity.expanding().mean().shift(1).reset_index(level=0, drop=True)
)
ordered_df["field_productivity_history_std"] = (
    field_productivity.expanding().std().shift(1).reset_index(level=0, drop=True)
)
ordered_df["field_productivity_history_min"] = (
    field_productivity.expanding().min().shift(1).reset_index(level=0, drop=True)
)
ordered_df["field_productivity_history_max"] = (
    field_productivity.expanding().max().shift(1).reset_index(level=0, drop=True)
)
ordered_df["field_productivity_history_trend"] = (
    ordered_df["field_prev_productivity"] - ordered_df["field_productivity_history_mean"]
)

df = ordered_df.sort_index()


# %%
drop_columns = {
    TARGET,
    "enrich_error",
    "cropwise_error",
    "weather_error",
    "soil_error",
    "sowing_date",
    "harvesting_date",
}

feature_columns = [column for column in df.columns if column not in drop_columns]
X = df[feature_columns].copy()
y = df[TARGET].astype(float)

for column in X.select_dtypes(include=["bool"]).columns:
    X[column] = X[column].astype(int)

categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
for column in categorical_columns:
    X[column] = X[column].fillna("__missing__").astype(str)

numeric_columns = [column for column in X.columns if column not in categorical_columns]
X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)

print(f"rows={len(df)} features={len(feature_columns)} categorical={len(categorical_columns)}")
print(f"categorical_columns={categorical_columns}")


# %%
train_idx, valid_idx = train_test_split(
    np.arange(len(df)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["year"] if df["year"].nunique() > 1 else None,
)

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]
y_train_raw = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]
y_train = np.log1p(y_train_raw)
y_valid_for_eval = np.log1p(y_valid)

cat_feature_indices = [X.columns.get_loc(column) for column in categorical_columns]

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
valid_pool = Pool(X_valid, y_valid_for_eval, cat_features=cat_feature_indices)

print(f"train={len(X_train)} valid={len(X_valid)}")


# %%
model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=2400,
    learning_rate=0.025,
    depth=4,
    l2_leaf_reg=2,
    random_strength=1.5,
    bagging_temperature=0.5,
    random_seed=RANDOM_STATE,
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=120,
    verbose=50,
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)


# %%
valid_pred = np.expm1(model.predict(valid_pool))
valid_pred = np.maximum(valid_pred, 0)
mae = mean_absolute_error(y_valid, valid_pred)
rmse = mean_squared_error(y_valid, valid_pred) ** 0.5
r2 = r2_score(y_valid, valid_pred)

metrics = pd.DataFrame(
    {
        "metric": ["MAE", "RMSE", "R2"],
        "value": [mae, rmse, r2],
    }
)
print(metrics.to_string(index=False))


# %%
prediction_sample = pd.DataFrame(
    {
        "field_id": df.iloc[valid_idx]["field_id"].to_numpy(),
        "year": df.iloc[valid_idx]["year"].to_numpy(),
        "y_true": y_valid.to_numpy(),
        "y_pred": valid_pred,
        "abs_error": np.abs(y_valid.to_numpy() - valid_pred),
    }
).sort_values("abs_error", ascending=False)

print(prediction_sample.head(20).to_string(index=False))


# %%
feature_importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "importance": model.get_feature_importance(train_pool),
    }
).sort_values("importance", ascending=False)

print(feature_importance.head(30).to_string(index=False))


# %%
MODEL_PATH = Path("catboost_yield_model.cbm")
IMPORTANCE_PATH = Path("catboost_feature_importance.csv")
PREDICTIONS_PATH = Path("catboost_valid_predictions.csv")

model.save_model(MODEL_PATH)
feature_importance.to_csv(IMPORTANCE_PATH, index=False)
prediction_sample.to_csv(PREDICTIONS_PATH, index=False)

print(f"saved_model={MODEL_PATH}")
print(f"saved_importance={IMPORTANCE_PATH}")
print(f"saved_predictions={PREDICTIONS_PATH}")
