# %%
from pathlib import Path

import numpy as np
import pandas as pd
from pycaret.regression import compare_models, predict_model, pull, setup
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# %%
DATA_PATH = Path("data/enriched_yield_dataset.csv")
TARGET = "productivity"
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = Path("pycaret_results")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    "enrich_error",
    "cropwise_error",
    "weather_error",
    "soil_error",
    "sowing_date",
    "harvesting_date",
}

model_df = df[[column for column in df.columns if column not in drop_columns]].copy()

for column in model_df.select_dtypes(include=["bool"]).columns:
    model_df[column] = model_df[column].astype(int)

categorical_columns = [
    column
    for column in model_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if column != TARGET
]

for column in categorical_columns:
    model_df[column] = model_df[column].fillna("__missing__").astype(str)

numeric_columns = [column for column in model_df.columns if column not in categorical_columns]
model_df[numeric_columns] = model_df[numeric_columns].replace([np.inf, -np.inf], np.nan)

train_idx, valid_idx = train_test_split(
    np.arange(len(model_df)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=model_df["year"] if model_df["year"].nunique() > 1 else None,
)

train_df = model_df.iloc[train_idx].copy()
valid_df = model_df.iloc[valid_idx].copy()

print(f"rows={len(model_df)} train={len(train_df)} valid={len(valid_df)}")
print(f"features={model_df.shape[1] - 1} categorical={len(categorical_columns)}")
print(f"categorical_columns={categorical_columns}")


# %%
setup(
    data=train_df,
    test_data=valid_df,
    target=TARGET,
    session_id=RANDOM_STATE,
    fold=5,
    categorical_features=categorical_columns,
    preprocess=True,
    verbose=False,
)


# %%
top_models = compare_models(
    sort="RMSE",
    n_select=10,
    turbo=True,
    verbose=False,
)

if not isinstance(top_models, list):
    top_models = [top_models]

cv_leaderboard = pull()
cv_leaderboard.to_csv(OUTPUT_DIR / "pycaret_cv_leaderboard.csv", index=False)
print(cv_leaderboard.to_string(index=False))


# %%
holdout_rows = []
prediction_frames = []

for rank, model in enumerate(top_models, start=1):
    predictions = predict_model(model, data=valid_df, verbose=False)
    prediction_column = "prediction_label"
    y_true = predictions[TARGET].astype(float)
    y_pred = predictions[prediction_column].astype(float)
    holdout_rows.append(
        {
            "rank": rank,
            "model": type(model).__name__,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred),
        }
    )
    frame = pd.DataFrame(
        {
            "rank": rank,
            "model": type(model).__name__,
            "field_id": valid_df["field_id"].to_numpy(),
            "year": valid_df["year"].to_numpy(),
            "y_true": y_true.to_numpy(),
            "y_pred": y_pred.to_numpy(),
            "abs_error": np.abs(y_true.to_numpy() - y_pred.to_numpy()),
        }
    )
    prediction_frames.append(frame)

holdout_leaderboard = pd.DataFrame(holdout_rows).sort_values("RMSE")
holdout_predictions = pd.concat(prediction_frames, ignore_index=True)

holdout_leaderboard.to_csv(OUTPUT_DIR / "pycaret_holdout_leaderboard.csv", index=False)
holdout_predictions.to_csv(OUTPUT_DIR / "pycaret_holdout_predictions.csv", index=False)

print(holdout_leaderboard.to_string(index=False))


# %%
best_rank = int(holdout_leaderboard.iloc[0]["rank"])
best_predictions = holdout_predictions[holdout_predictions["rank"] == best_rank]
best_predictions = best_predictions.sort_values("abs_error", ascending=False)
best_predictions.to_csv(OUTPUT_DIR / "pycaret_best_model_predictions.csv", index=False)

print(best_predictions.head(20).to_string(index=False))
print(f"saved_dir={OUTPUT_DIR}")
