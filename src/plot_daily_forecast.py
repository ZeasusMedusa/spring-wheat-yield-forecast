# %%
from pathlib import Path

import ast
import json

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split


# %%
DATA_PATH = Path("data/enriched_yield_dataset.csv")
SOURCE_PATH = Path("productivity_data_v2.csv")
MODEL_PATH = Path("catboost_yield_model.cbm")
FIGURE_DIR = Path("reports/yield_model_report/figures")
OUTPUT_DIR = Path("reports/yield_model_report/tables")
FIGURE_PATH = FIGURE_DIR / "daily_forecast_8_validation_fields_subplots.png"
TABLE_PATH = OUTPUT_DIR / "daily_forecast_8_validation_fields_subplots.csv"
TARGET = "productivity"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FIELDS = 8
LOCAL_SERIES_COLUMNS = [
    "ndvi",
    "ndvi_l8",
    "ndvi_ps8",
    "ndvi_s2a",
    "ndvi_s2b",
    "ndvi_s2c",
    "soil_moisture",
    "temperature",
]
NDVI_COLUMNS = {"ndvi", "ndvi_l8", "ndvi_ps8", "ndvi_s2a", "ndvi_s2b", "ndvi_s2c"}

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
def add_training_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    return ordered_df.sort_index()


def model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
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
    for column in X.select_dtypes(include=["bool"]).columns:
        X[column] = X[column].astype(int)
    categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return X, feature_columns, categorical_columns


def prepare_for_model(X: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    X = X.copy()
    for column in categorical_columns:
        X[column] = X[column].fillna("__missing__").astype(str)
    numeric_columns = [column for column in X.columns if column not in categorical_columns]
    X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return X


# %%
def parse_series(value: object) -> list[tuple[pd.Timestamp, float]]:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
    else:
        parsed = value
    result = []
    if not isinstance(parsed, list):
        return result
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        date = pd.to_datetime(item[0], errors="coerce")
        if pd.isna(date):
            continue
        raw_value = item[1]
        if isinstance(raw_value, list):
            numbers = [float(x) for x in raw_value if pd.notna(x)]
            if not numbers:
                continue
            result.append((date.normalize(), float(np.mean(numbers))))
        elif pd.notna(raw_value):
            result.append((date.normalize(), float(raw_value)))
    return result


def trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.asarray(values, dtype=float), 1)[0])


def prefix_stats(series: list[tuple[pd.Timestamp, float]], current_date: pd.Timestamp, column: str) -> dict[str, float | int]:
    values = [value for date, value in series if date <= current_date]
    prefix = f"local_{column}"
    if not values:
        return {
            f"{prefix}_missing": 1,
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_first": np.nan,
            f"{prefix}_last": np.nan,
            f"{prefix}_trend": np.nan,
            f"{prefix}_is_ndvi_like": int(column in NDVI_COLUMNS),
        }
    return {
        f"{prefix}_missing": 0,
        f"{prefix}_count": len(values),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_first": float(values[0]),
        f"{prefix}_last": float(values[-1]),
        f"{prefix}_trend": trend(values),
        f"{prefix}_is_ndvi_like": int(column in NDVI_COLUMNS),
    }


def build_series_by_column(source_df: pd.DataFrame, base_row: pd.Series) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    source_match = source_df[
        (source_df["field_id"].astype(str) == str(base_row["field_id"]))
        & (source_df["year"].astype(int) == int(base_row["year"]))
        & (source_df["crop_name"].astype(str) == str(base_row["crop_name"]))
    ]
    result: dict[str, list[tuple[pd.Timestamp, float]]] = {column: [] for column in LOCAL_SERIES_COLUMNS}
    for _, source_row in source_match.iterrows():
        column = str(source_row["historical_values.product_type"])
        if column in result:
            result[column] = parse_series(source_row["historical_values.value"])
    return result


def select_validation_rows(valid_scores: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    available_keys = set(
        zip(
            source_df["field_id"].astype(str),
            source_df["year"].astype(int),
            source_df["crop_name"].astype(str),
        )
    )
    candidates = valid_scores[
        valid_scores.apply(
            lambda row: (str(row["field_id"]), int(row["year"]), str(row["crop_name"])) in available_keys,
            axis=1,
        )
    ].copy()
    candidates["harvesting_date"] = pd.to_datetime(candidates["harvesting_date"], errors="coerce")
    candidates = candidates.dropna(subset=["harvesting_date"]).sort_values("abs_error")

    unique_candidates = candidates.drop_duplicates("field_id", keep="first").reset_index(drop=True)
    if len(unique_candidates) < N_FIELDS:
        raise RuntimeError(f"Need {N_FIELDS} distinct validation fields, got {len(unique_candidates)}")

    by_end_date = unique_candidates.sort_values("harvesting_date").reset_index(drop=True)
    best_start = 0
    best_span = pd.Timedelta.max
    for start in range(0, len(by_end_date) - N_FIELDS + 1):
        window = by_end_date.iloc[start : start + N_FIELDS]
        span = window["harvesting_date"].max() - window["harvesting_date"].min()
        if span < best_span:
            best_span = span
            best_start = start
    return by_end_date.iloc[best_start : best_start + N_FIELDS].sort_values("harvesting_date").reset_index(drop=True)


def daily_predictions_for_row(
    base_row: pd.Series,
    source_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    cat_feature_indices: list[int],
    model: CatBoostRegressor,
) -> pd.DataFrame:
    series_by_column = build_series_by_column(source_df, base_row)

    sowing = pd.to_datetime(base_row["sowing_date"]).normalize()
    harvesting = pd.to_datetime(base_row["harvesting_date"]).normalize()
    dates = pd.date_range(sowing, harvesting, freq="D")
    season_length = max((harvesting - sowing).days + 1, 1)

    daily_rows = []
    sum_columns = [column for column in feature_columns if column.startswith("season_weather_") and column.endswith("_sum")]
    day_columns = [column for column in feature_columns if column.startswith("season_weather_") and column.endswith("_days")]

    for current_date in dates:
        progress_days = (current_date - sowing).days + 1
        progress = progress_days / season_length
        daily = base_row[feature_columns].copy()
        daily["season_duration_days"] = progress_days
        daily["harvesting_month"] = current_date.month
        daily["harvesting_dayofyear"] = current_date.dayofyear
        daily["season_mid_dayofyear"] = (daily["sowing_dayofyear"] + daily["harvesting_dayofyear"]) / 2
        daily["season_weather_days"] = progress_days
        for column in sum_columns + day_columns:
            if pd.notna(daily[column]):
                daily[column] = daily[column] * progress
        for column, series in series_by_column.items():
            for key, value in prefix_stats(series, current_date, column).items():
                if key in daily.index:
                    daily[key] = value
        daily_rows.append(daily)

    daily_X = prepare_for_model(pd.DataFrame(daily_rows, columns=feature_columns), categorical_columns)
    daily_pred = np.maximum(np.expm1(model.predict(Pool(daily_X, cat_features=cat_feature_indices))), 0)
    return pd.DataFrame(
        {
            "date": dates,
            "day_from_sowing": np.arange(1, len(dates) + 1),
            "field_id": int(base_row["field_id"]),
            "year": int(base_row["year"]),
            "prediction": daily_pred,
            "actual_productivity": float(base_row[TARGET]),
        }
    )


# %%
df = add_training_features(pd.read_csv(DATA_PATH))
X, feature_columns, categorical_columns = model_frame(df)
X_prepared = prepare_for_model(X, categorical_columns)

train_idx, valid_idx = train_test_split(
    np.arange(len(df)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["year"] if df["year"].nunique() > 1 else None,
)

cat_feature_indices = [feature_columns.index(column) for column in categorical_columns]
model = CatBoostRegressor()
model.load_model(MODEL_PATH)
valid_pred = np.maximum(np.expm1(model.predict(Pool(X_prepared.iloc[valid_idx], cat_features=cat_feature_indices))), 0)

valid_scores = pd.DataFrame(
    {
        "row_index": valid_idx,
        "field_id": df.iloc[valid_idx]["field_id"].to_numpy(),
        "field_name": df.iloc[valid_idx]["field_name"].to_numpy(),
        "crop_name": df.iloc[valid_idx]["crop_name"].to_numpy(),
        "year": df.iloc[valid_idx]["year"].to_numpy(),
        "harvesting_date": df.iloc[valid_idx]["harvesting_date"].to_numpy(),
        "y_true": df.iloc[valid_idx][TARGET].to_numpy(),
        "y_pred": valid_pred,
        "abs_error": np.abs(df.iloc[valid_idx][TARGET].to_numpy() - valid_pred),
    }
)

source_df = pd.read_csv(SOURCE_PATH)
selected = select_validation_rows(valid_scores, source_df)

plot_frames = []
for row in selected.itertuples(index=False):
    base_row = df.iloc[int(row.row_index)].copy()
    plot_frames.append(
        daily_predictions_for_row(base_row, source_df, feature_columns, categorical_columns, cat_feature_indices, model)
    )

plot_df = pd.concat(plot_frames, ignore_index=True)
plot_df.to_csv(TABLE_PATH, index=False)


# %%
colors = plt.cm.tab10(np.linspace(0, 1, N_FIELDS))
plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 11})
fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=False, sharey=True)
axes_flat = axes.ravel()
y_values = pd.concat([plot_df["prediction"], plot_df["actual_productivity"]], ignore_index=True)
y_span = float(y_values.max() - y_values.min())
y_margin = max(y_span * 0.08, 2.0)
y_min = float(y_values.min() - y_margin)
y_max = float(y_values.max() + y_margin)

for subplot_index, ax in enumerate(axes_flat):
    pair = selected.iloc[subplot_index * 2 : subplot_index * 2 + 2]
    for pair_offset, (_, selected_row) in enumerate(pair.iterrows()):
        color = colors[subplot_index * 2 + pair_offset]
        field_id = int(selected_row["field_id"])
        year = int(selected_row["year"])
        field_df = plot_df[(plot_df["field_id"] == field_id) & (plot_df["year"] == year)].copy()
        label = f"field_id={field_id}, {year}"
        ax.plot(
            field_df["day_from_sowing"],
            field_df["prediction"],
            color=color,
            linewidth=2.2,
            alpha=0.96,
            zorder=3,
            label=f"Прогноз: {label}",
        )
        actual_line = ax.axhline(
            y=float(field_df["actual_productivity"].iloc[0]),
            color=color,
            linestyle="--",
            linewidth=2.6,
            alpha=1.0,
            zorder=6,
            label=f"Факт: {label}",
        )
        actual_line.set_path_effects([path_effects.Stroke(linewidth=4.0, foreground="white"), path_effects.Normal()])
    end_dates = pd.to_datetime(pair["harvesting_date"]).dt.strftime("%Y-%m-%d").tolist()
    ax.set_title(f"Поля {subplot_index * 2 + 1}–{subplot_index * 2 + len(pair)}; уборка: {', '.join(end_dates)}")
    ax.set_xlabel("День от даты сева")
    ax.set_ylabel("Урожайность, ц/га")
    ax.grid(alpha=0.25)
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8.5, frameon=True, loc="best")

fig.suptitle("Ежедневные прогнозы модели: 8 полей validation split с близкими датами уборки", fontsize=16, y=0.995)
fig.tight_layout()
fig.savefig(FIGURE_PATH, bbox_inches="tight")
plt.close(fig)

print(f"selected_fields={selected[['field_id', 'year', 'harvesting_date', 'y_true', 'y_pred', 'abs_error']].to_dict('records')}")
print(f"saved_figure={FIGURE_PATH}")
print(f"saved_table={TABLE_PATH}")
print(f"rows={len(plot_df)}")
