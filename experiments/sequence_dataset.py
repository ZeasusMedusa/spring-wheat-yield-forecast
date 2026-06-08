import ast
import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class YieldForecastDataset(Dataset):
    """
    Dataset для прогноза урожайности по временным признакам.

    Идея:
    - Один sample = одно поле в конкретный день после sowing_date.
    - Признаки считаются только по данным <= current_date.
    - Для каждого product_type считаются:
        latest, mean/std/min/max/sum/count за окна.
    - Для каждого признака есть mask: 1 если данные были, 0 если данных нет.
    - train/val split делается по группам field_id + year, чтобы не было leakage.

    Пример использования:
        df = pd.read_csv("fields.csv")
        train_ds = YieldForecastDataset(df, crop_name="Чечевица", split="train")
        val_ds   = YieldForecastDataset(df, crop_name="Чечевица", split="val",
                                        product_types=train_ds.product_types,
                                        feature_names=train_ds.feature_names)

        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        batch = next(iter(loader))
        # batch["x"]    — (B, F) float32
        # batch["mask"] — (B, F) float32  (1 = есть данные, 0 = нет)
        # batch["y"]    — (B, 1) float32  урожайность
        # batch["meta"] — dict с полезной мета-информацией
    """

    def __init__(
        self,
        df: pd.DataFrame,
        crop_name: str,
        split: Literal["train", "val"] = "train",
        val_size: float = 0.2,
        seed: int = 42,
        step_days: int = 7,
        min_day_from_sowing: int = 0,
        max_day_from_sowing: int | None = None,
        windows: tuple[int, ...] = (7, 30, 90, 180, 365),
        product_types: list[str] | None = None,
        feature_names: list[str] | None = None,
        stats: tuple[str, ...] = ("mean", "std", "min", "max", "sum", "count"),
        fill_value: float = 0.0,
        group_cols: tuple[str, ...] = ("field_id", "year"),
        year_mean: dict[int, float] | None = None,
    ) -> None:
        """
        Args:
            df:                   Исходный датафрейм.
            crop_name:            Название культуры (фильтр по колонке crop_name).
            split:                "train" или "val".
            val_size:             Доля групп в val (0..1).
            seed:                 Seed для детерминированного сплита.
            step_days:            Шаг по дням при генерации samples.
            min_day_from_sowing:  С какого дня от sowing_date начинать samples.
            max_day_from_sowing:  До какого дня (включительно). None = до harvesting_date.
            windows:              Окна агрегации в днях.
            product_types:        Список product_type для признаков.
                                  None = авто из данных.
                                  Передавай train_ds.product_types в val, чтобы признаки совпадали.
            feature_names:        Фиксированный порядок признаков.
                                  None = строится автоматически.
                                  Передавай train_ds.feature_names в val.
            stats:                Статистики для агрегации по окнам.
            fill_value:           Значение для отсутствующих признаков (маска = 0).
            group_cols:           Колонки для группировки (обычно field_id + year).
        """
        if split not in {"train", "val"}:
            raise ValueError("split должен быть 'train' или 'val'")

        self.crop_name = crop_name
        self.split = split
        self.val_size = val_size
        self.seed = seed
        self.step_days = step_days
        self.min_day_from_sowing = min_day_from_sowing
        self.max_day_from_sowing = max_day_from_sowing
        self.windows = windows
        self.stats = stats
        self.fill_value = float(fill_value)
        self.group_cols = group_cols

        if year_mean is not None:
            self.year_mean = year_mean  # override из train

        self.df = self._prepare_df(df)
        self.df = self.df[self.df["crop_name"] == crop_name].copy()

        if self.df.empty:
            raise ValueError(f"Нет строк для crop_name={crop_name!r}")

        if product_types is None:
            self.product_types = sorted(
                self.df["historical_values.product_type"].dropna().unique().tolist()
            )
        else:
            self.product_types = product_types

        self.groups = self._build_groups(self.df)
        self._enrich_with_prev_year(self.groups)
        self.groups = self._filter_groups_by_split(self.groups, split)

        if not self.groups:
            raise ValueError(
                f"После split={split!r} не осталось групп. "
                f"Проверь crop_name={crop_name!r}, val_size={val_size}, group_cols={group_cols}."
            )

        # Если feature_names переданы снаружи (например из train) — используем их.
        # Это гарантирует совпадение признаков между train и val.
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = self._build_feature_names(self.groups)

        self.samples = self._build_samples(self.groups)

        self.year_mean = self._build_year_mean(self.groups)


        if not self.samples:
            raise ValueError(
                "Не получилось создать samples. "
                "Проверь sowing_date, harvesting_date, step_days, min/max_day_from_sowing."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        group = self.groups[sample["group_key"]]
        current_date = sample["current_date"]
        day_from_sowing = sample["day_from_sowing"]

        total_days = max(1, int((group["harvesting_date"] - group["sowing_date"]).days))
        season_progress = day_from_sowing / total_days

        features = self._make_features_for_date(
            series=group["series"],
            current_date=current_date,
            day_from_sowing=day_from_sowing,
            season_progress=season_progress,
        )

        x = np.array([features[name][0] for name in self.feature_names], dtype=np.float32)
        mask = np.array([features[name][1] for name in self.feature_names], dtype=np.float32)
        year_mean = self.year_mean.get(group["year"], float("nan"))

        return {
            "x": torch.from_numpy(x),
            "year": torch.tensor([float(group["year"])], dtype=torch.float32),
            "year_mean": torch.tensor([year_mean], dtype=torch.float32),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor([group["productivity"]], dtype=torch.float32),
            "meta": {
                "field_id": str(group["field_id"]),
                "field_name": str(group["field_name"]) if group["field_name"] is not None else "",
                "crop_name": group["crop_name"],
                "year": group["year"],
                "sowing_date": str(group["sowing_date"].date()),
                "harvesting_date": str(group["harvesting_date"].date()),
                "current_date": str(current_date.date()),
                "day_from_sowing": day_from_sowing,
                "season_progress": season_progress,
                "productivity": group["productivity"],
                "prev_year_productivity": group["prev_year_productivity"],
                "prev_year_crop": group["prev_year_crop"],
            },
        }

    # ------------------------------------------------------------------
    # DataFrame preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        required = {
            "field_id",
            "crop_name",
            "year",
            "sowing_date",
            "harvesting_date",
            "productivity",
            "historical_values.product_type",
            "historical_values.value",
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"В dataframe нет колонок: {sorted(missing)}")

        out = df.copy()
        out["sowing_date"] = pd.to_datetime(out["sowing_date"], errors="coerce")
        out["harvesting_date"] = pd.to_datetime(out["harvesting_date"], errors="coerce")
        out["productivity"] = pd.to_numeric(out["productivity"], errors="coerce")

        out = out.dropna(
            subset=[
                "field_id",
                "crop_name",
                "year",
                "sowing_date",
                "harvesting_date",
                "productivity",
                "historical_values.product_type",
                "historical_values.value",
            ]
        )

        return out

    @staticmethod
    def _build_year_mean(groups: dict[str, dict[str, Any]]) -> dict[int, float]:
        """Средняя урожайность по году — считается только из групп этого сплита."""
        from collections import defaultdict
        sums: dict[int, list[float]] = defaultdict(list)
        for group in groups.values():
            sums[group["year"]].append(group["productivity"])
        return {year: float(np.mean(vals)) for year, vals in sums.items()}

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_history(value: Any) -> list[tuple[pd.Timestamp, Any]]:
        """
        Парсит historical_values.value.

        Ожидаемый формат:
            [["2024-05-01", 0.253], ["2024-05-02", 0.247], ...]
            или
            [["2024-05-01", [0.1, 0.2, 0.3, 0.4]], ...]  — для векторных признаков

        value может быть строкой, list или уже распарсенным объектом.
        """
        try:
            raw = ast.literal_eval(value) if isinstance(value, str) else value
        except Exception:
            return []

        if not isinstance(raw, list):
            return []

        parsed = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue

            date_raw, val_raw = item
            date = pd.to_datetime(date_raw, errors="coerce")

            if pd.isna(date):
                continue

            parsed.append((date.normalize(), val_raw))

        return parsed

    @staticmethod
    def _to_float_array(value: Any) -> np.ndarray:
        """
        Приводит scalar/list к np.ndarray[float32], убирая нечисловые значения.

        Примеры:
            0.25                -> [0.25]
            [0.1, 0.2, 0.3]    -> [0.1, 0.2, 0.3]
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=np.float32).reshape(-1)
        else:
            arr = np.array([value], dtype=np.float32)

        return arr[np.isfinite(arr)]

    # ------------------------------------------------------------------
    # Group building
    # ------------------------------------------------------------------

    def _build_groups(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """
        Группирует строки по group_cols и строит словарь:
            group_key -> {field_id, field_name, crop_name, year,
                          sowing_date, harvesting_date, productivity,
                          series: {product_type -> DataFrame(date, dim, value)}}
        """
        has_field_name = "field_name" in df.columns
        groups: dict[str, dict[str, Any]] = {}

        for group_values, group_df in df.groupby(list(self.group_cols), dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            first = group_df.iloc[0]
            group_key = self._make_group_key(group_values)

            series: dict[str, pd.DataFrame] = {}

            for product_type, pt_df in group_df.groupby("historical_values.product_type"):
                if product_type not in self.product_types:
                    continue

                records = []
                for raw_history in pt_df["historical_values.value"]:
                    for date, raw_value in self._safe_parse_history(raw_history):
                        for dim_idx, val in enumerate(self._to_float_array(raw_value)):
                            records.append({"date": date, "dim": dim_idx, "value": float(val)})

                if records:
                    series[str(product_type)] = (
                        pd.DataFrame(records)
                        .dropna(subset=["date", "dim", "value"])
                        .sort_values("date")
                        .reset_index(drop=True)
                    )

            if not series:
                continue

            groups[group_key] = {
                "group_key": group_key,
                "field_id": first["field_id"],
                "field_name": first["field_name"] if has_field_name else None,
                "crop_name": first["crop_name"],
                "year": int(first["year"]),
                "sowing_date": first["sowing_date"].normalize(),
                "harvesting_date": first["harvesting_date"].normalize(),
                "productivity": float(first["productivity"]),
                "series": series,
            }

        return groups

    def _enrich_with_prev_year(self, groups: dict[str, dict[str, Any]]) -> None:
        """
        Добавляет в каждую группу урожайность и культуру предыдущего года
        того же поля. Если предыдущего года нет — None.
        """
        # field_id -> {year -> group}
        by_field: dict[Any, dict[int, dict]] = {}
        for group in groups.values():
            by_field.setdefault(group["field_id"], {})[group["year"]] = group

        for group in groups.values():
            prev = by_field[group["field_id"]].get(group["year"] - 1)
            group["prev_year_productivity"] = prev["productivity"] if prev else None
            group["prev_year_crop"] = prev["crop_name"] if prev else None

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _filter_groups_by_split(
        self,
        groups: dict[str, dict[str, Any]],
        split: Literal["train", "val"],
    ) -> dict[str, dict[str, Any]]:
        is_val = split == "val"
        selected_keys = [
            key
            for key in sorted(groups.keys())
            if (self._stable_random_01(key, self.seed) < self.val_size) == is_val
        ]
        return {key: groups[key] for key in selected_keys}

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def _build_samples(self, groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Для каждой группы генерирует список samples — по одному на каждый
        шаг step_days от min_day_from_sowing до harvesting_date.
        """
        samples = []

        for group_key, group in groups.items():
            sowing = group["sowing_date"]
            harvesting = group["harvesting_date"]

            if harvesting <= sowing:
                continue

            total_days = int((harvesting - sowing).days)
            end_day = total_days if self.max_day_from_sowing is None else min(total_days, self.max_day_from_sowing)

            for day in range(self.min_day_from_sowing, end_day + 1, self.step_days):
                samples.append(
                    {
                        "group_key": group_key,
                        "current_date": sowing + pd.Timedelta(days=day),
                        "day_from_sowing": day,
                    }
                )

        return samples

    # ------------------------------------------------------------------
    # Feature schema
    # ------------------------------------------------------------------

    def _build_feature_names(self, groups: dict[str, dict[str, Any]]) -> list[str]:
        """
        Строит упорядоченный список имён признаков.

        Структура имени:
            meta__day_from_sowing
            meta__season_progress
            {product_type}__dim{N}__latest
            {product_type}__dim{N}__latest_age_days
            {product_type}__dim{N}__w{W}__{stat}
        """
        # Собираем dims для каждого product_type по всем группам
        dims_by_product: dict[str, set[int]] = {pt: set() for pt in self.product_types}
        for group in groups.values():
            for pt, frame in group["series"].items():
                if pt in dims_by_product:
                    dims_by_product[pt].update(int(d) for d in frame["dim"].unique())

        names: list[str] = ["meta__day_from_sowing", "meta__season_progress"]

        for pt in self.product_types:
            for dim in sorted(dims_by_product.get(pt, {0})):
                base = f"{pt}__dim{dim}"
                names.append(f"{base}__latest")
                names.append(f"{base}__latest_age_days")
                for window in self.windows:
                    for stat in self.stats:
                        names.append(f"{base}__w{window}__{stat}")

        return names

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _make_features_for_date(
        self,
        series: dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
        day_from_sowing: int,
        season_progress: float,
    ) -> dict[str, tuple[float, float]]:
        """
        Возвращает словарь {feature_name: (value, mask)}.

        mask = 1.0  — данные есть
        mask = 0.0  — данных нет, value = fill_value
        """
        # Инициализируем все признаки fill_value с маской 0
        features: dict[str, tuple[float, float]] = {
            name: (self.fill_value, 0.0) for name in self.feature_names
        }

        # Мета-признаки всегда известны
        features["meta__day_from_sowing"] = (float(day_from_sowing), 1.0)
        features["meta__season_progress"] = (float(season_progress), 1.0)

        for pt, frame in series.items():
            if pt not in self.product_types:
                continue

            for dim in sorted(int(d) for d in frame["dim"].unique()):
                base = f"{pt}__dim{dim}"

                dim_frame = frame[frame["dim"] == dim].sort_values("date")
                known = dim_frame[dim_frame["date"] <= current_date]

                # latest + latest_age
                if not known.empty:
                    latest_row = known.iloc[-1]
                    key_latest = f"{base}__latest"
                    key_age = f"{base}__latest_age_days"
                    if key_latest in features:
                        features[key_latest] = (float(latest_row["value"]), 1.0)
                    if key_age in features:
                        age = int((current_date - latest_row["date"]).days)
                        features[key_age] = (float(age), 1.0)

                # Оконные агрегаты
                for window in self.windows:
                    start_date = current_date - pd.Timedelta(days=window)
                    window_values = dim_frame[
                        (dim_frame["date"] > start_date) & (dim_frame["date"] <= current_date)
                    ]["value"].to_numpy(dtype=np.float32)

                    if window_values.size == 0:
                        continue

                    stats = self._calc_stats(window_values)
                    for stat, val in stats.items():
                        key = f"{base}__w{window}__{stat}"
                        if key in features:
                            features[key] = (val, 1.0)

        return features

    # ------------------------------------------------------------------
    # Stat helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sum": float(np.sum(values)),
            "count": float(values.size),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _make_group_key(group_values: tuple[Any, ...]) -> str:
        return "__".join(str(v) for v in group_values)

    @staticmethod
    def _stable_random_01(value: str, seed: int) -> float:
        """Детерминированный хэш строки в [0, 1) для стабильного сплита."""
        raw = f"{seed}::{value}".encode("utf-8")
        digest = hashlib.md5(raw).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF