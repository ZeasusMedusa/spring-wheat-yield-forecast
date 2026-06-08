#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


Json = dict[str, Any]

CROPWISE_BASE_URL = "https://operations.cropwise.com/api/v3"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
SOILGRIDS_QUERY_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

CROPWISE_TOKEN_ENVS = (
    "CROPWISE_USER_API_TOKEN",
    "USER_API_TOKEN",
    "X_USER_API_TOKEN",
    "CROPWISE_TOKEN",
)
VALID_AUTH_HEADERS = ("x-user-api-token", "bearer")
REQUIRED_COLUMNS = ("field_id", "sowing_date", "harvesting_date")
LOCAL_SERIES_COLUMNS = (
    "ndvi",
    "ndvi_l8",
    "ndvi_ps8",
    "ndvi_s2a",
    "ndvi_s2b",
    "ndvi_s2c",
    "soil_moisture",
    "temperature",
)
NDVI_LIKE_COLUMNS = frozenset(
    ("ndvi", "ndvi_l8", "ndvi_ps8", "ndvi_s2a", "ndvi_s2b", "ndvi_s2c")
)
LAT_COLUMNS = ("lat", "latitude", "centroid_lat", "field_lat", "field_centroid_lat")
LON_COLUMNS = ("lon", "lng", "longitude", "centroid_lon", "field_lon", "field_centroid_lon")
AREA_COLUMNS = ("field_area", "area", "area_ha", "field_area_ha", "area_hectares")
GEOMETRY_COLUMNS = ("geometry", "geojson", "field_geometry", "boundary", "coordinates")

WEATHER_DAILY_VARS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
)
SOIL_PROPERTIES = (
    "bdod",
    "cec",
    "clay",
    "nitrogen",
    "phh2o",
    "sand",
    "silt",
    "soc",
)
SOIL_DEPTHS = ("0-5cm", "5-15cm", "15-30cm")
GDD_BASE_C = 5.0
HEAT_THRESHOLD_C = 30.0
DRY_PRECIP_MM = 1.0


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    cache_dir: Path
    output_format: str
    cropwise_base_url: str
    cropwise_token: str | None
    cropwise_token_source: str | None
    cropwise_auth_header: str
    max_rows: int | None
    sleep_s: float
    request_timeout_s: int
    skip_cropwise: bool
    skip_weather: bool
    skip_soil: bool
    fail_fast: bool
    diagnose_cropwise: bool


@dataclass
class SourceUsage:
    cropwise_rows: int = 0
    weather_rows: int = 0
    soil_rows: int = 0
    local_ts_rows: int = 0
    error_rows: int = 0


class CropwiseAuthError(RuntimeError):
    pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dotenv_if_present(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_cropwise_token(env_name: str | None) -> tuple[str | None, str | None]:
    if env_name:
        value = os.getenv(env_name)
        return (value.strip(), env_name) if value and value.strip() else (None, env_name)
    for candidate in CROPWISE_TOKEN_ENVS:
        value = os.getenv(candidate)
        if value and value.strip():
            return value.strip(), candidate
    return None, None


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()


def safe_json_loads(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        numbers = [_coerce_numeric(item) for item in value]
        clean = [item for item in numbers if item is not None]
        return float(sum(clean) / len(clean)) if clean else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def load_source_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_rows)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    return df


def parse_historical_series(value: Any) -> pd.DataFrame:
    raw = safe_json_loads(value)
    if not isinstance(raw, list):
        return pd.DataFrame(columns=pd.Index(["date", "value"]))
    rows: list[Json] = []
    for item in raw:
        when: Any
        val: Any
        if isinstance(item, list) and len(item) >= 2:
            when, val = item[0], item[1]
        elif isinstance(item, dict):
            when = item.get("date") or item.get("time")
            val = item.get("value")
        else:
            continue
        numeric = _coerce_numeric(val)
        parsed = pd.to_datetime(when, errors="coerce")
        if numeric is not None and not pd.isna(parsed):
            rows.append({"date": parsed, "value": numeric})
    return pd.DataFrame(rows, columns=pd.Index(["date", "value"])).dropna()


def _trend_slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    n = float(len(values))
    xs = [float(index) for index in range(len(values))]
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom


def aggregate_local_timeseries(row: pd.Series, start: date | None, end: date | None) -> Json:
    features: Json = {}
    for column in LOCAL_SERIES_COLUMNS:
        prefix = f"local_{column}"
        if column not in row.index:
            features[f"{prefix}_missing"] = True
            continue
        series_rows: list[tuple[date, float]] = []
        for item in parse_historical_series(row[column]).to_dict("records"):
            when = parse_date(item["date"])
            if when is None:
                continue
            if start is not None and when < start:
                continue
            if end is not None and when > end:
                continue
            series_rows.append((when, float(item["value"])))
        values = [value for _, value in sorted(series_rows, key=lambda item: item[0])]
        features[f"{prefix}_missing"] = len(values) == 0
        features[f"{prefix}_is_ndvi_like"] = column in NDVI_LIKE_COLUMNS
        if not values:
            for name in ("count", "mean", "std", "min", "max", "first", "last", "trend"):
                features[f"{prefix}_{name}"] = 0 if name == "count" else None
            continue
        value_series = pd.Series(values, dtype="float64")
        features.update(
            {
                f"{prefix}_count": int(value_series.count()),
                f"{prefix}_mean": float(value_series.mean()),
                f"{prefix}_std": float(value_series.std(ddof=0)) if len(values) > 1 else 0.0,
                f"{prefix}_min": float(value_series.min()),
                f"{prefix}_max": float(value_series.max()),
                f"{prefix}_first": float(values[0]),
                f"{prefix}_last": float(values[-1]),
                f"{prefix}_trend": _trend_slope(values),
            }
        )
    return features


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def cached_json(cache_path: Path, fetch: Any) -> Any:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Ignoring corrupt cache file: %s", cache_path)
    ensure_dir(cache_path.parent)
    data = fetch()
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(cache_path)
    return data


def request_json(
    session: requests.Session,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    timeout: int = 60,
) -> Any:
    response = session.get(url, headers=dict(headers or {}), params=params, timeout=timeout)
    if response.status_code in (401, 403):
        raise CropwiseAuthError(f"HTTP {response.status_code}")
    response.raise_for_status()
    return response.json()


def cropwise_headers(config: Config) -> dict[str, str]:
    token = config.cropwise_token or ""
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if config.cropwise_auth_header == "bearer":
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers["X-User-Api-Token"] = token
    return headers


def fetch_cropwise_field(
    session: requests.Session,
    config: Config,
    cache: dict[str, Json],
    field_id: str,
) -> Json:
    if config.skip_cropwise or not config.cropwise_token:
        return {}
    if field_id in cache:
        return cache[field_id]
    cache_path = config.cache_dir / "cropwise_fields" / f"{field_id}.json"

    def fetch() -> Any:
        time.sleep(config.sleep_s)
        return request_json(
            session,
            f"{config.cropwise_base_url.rstrip('/')}/fields/{field_id}",
            headers=cropwise_headers(config),
            timeout=config.request_timeout_s,
        )

    try:
        data = cached_json(cache_path, fetch)
    except CropwiseAuthError as exc:
        raise CropwiseAuthError(f"Cropwise auth failed ({exc}) for field {field_id}") from exc
    if isinstance(data, dict):
        cache[field_id] = data
        return data
    cache[field_id] = {}
    return {}


def _iter_values(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_values(value)


def extract_field_geometry(field_json: Mapping[str, Any]) -> Json | None:
    for candidate in _iter_values(field_json):
        if isinstance(candidate, dict) and candidate.get("type") in {"Point", "Polygon", "MultiPolygon"} and "coordinates" in candidate:
            return dict(candidate)
        if isinstance(candidate, dict):
            for key in ("shape_simplified_geojson", "shape_geojson", "geojson", "geometry"):
                parsed = safe_json_loads(candidate.get(key))
                if isinstance(parsed, dict) and parsed.get("type") in {"Point", "Polygon", "MultiPolygon"} and "coordinates" in parsed:
                    return parsed
    return None


def field_area_from_cropwise(field_json: Mapping[str, Any]) -> float | None:
    for candidate in _iter_values(field_json):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("calculated_area", "tillable_area", "legal_area", "area", "area_ha"):
            value = _coerce_numeric(candidate.get(key))
            if value is not None:
                return value
    return None


def _flatten_coordinates(coords: Any) -> list[tuple[float, float]]:
    if (
        isinstance(coords, Sequence)
        and not isinstance(coords, (str, bytes, bytearray))
        and len(coords) >= 2
        and isinstance(coords[0], (int, float))
        and isinstance(coords[1], (int, float))
    ):
        return [(float(coords[1]), float(coords[0]))]
    points: list[tuple[float, float]] = []
    if isinstance(coords, Sequence) and not isinstance(coords, (str, bytes, bytearray)):
        for item in coords:
            points.extend(_flatten_coordinates(item))
    return points


def compute_centroid(geometry: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    if not geometry:
        return None, None
    points = _flatten_coordinates(geometry.get("coordinates"))
    if not points:
        return None, None
    return sum(lat for lat, _ in points) / len(points), sum(lon for _, lon in points) / len(points)


def _polygon_area_ha(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    lat0 = sum(lat for lat, _ in points) / len(points)
    meters_per_lat = 111_320.0
    meters_per_lon = 111_320.0 * math.cos(math.radians(lat0))
    xy = [(lon * meters_per_lon, lat * meters_per_lat) for lat, lon in points]
    area = 0.0
    for index, (x1, y1) in enumerate(xy):
        x2, y2 = xy[(index + 1) % len(xy)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0 / 10_000.0


def _area_from_geometry(geometry: Mapping[str, Any] | None) -> float | None:
    if not geometry:
        return None
    points = _flatten_coordinates(geometry.get("coordinates"))
    return _polygon_area_ha(points)


def _numeric_from_row(row: pd.Series, columns: Sequence[str]) -> float | None:
    for column in columns:
        if column in row.index:
            value = _coerce_numeric(row[column])
            if value is not None:
                return value
    return None


def coords_from_row(row: pd.Series) -> tuple[float | None, float | None]:
    lat = _numeric_from_row(row, LAT_COLUMNS)
    lon = _numeric_from_row(row, LON_COLUMNS)
    if lat is not None and lon is not None:
        return lat, lon
    for column in GEOMETRY_COLUMNS:
        if column in row.index:
            geometry = safe_json_loads(row[column])
            if isinstance(geometry, dict):
                return compute_centroid(geometry)
    return None, None


def area_from_row(row: pd.Series) -> float | None:
    area = _numeric_from_row(row, AREA_COLUMNS)
    if area is not None:
        return area
    for column in GEOMETRY_COLUMNS:
        if column in row.index:
            geometry = safe_json_loads(row[column])
            if isinstance(geometry, dict):
                return _area_from_geometry(geometry)
    return None


def fetch_weather(session: requests.Session, config: Config, lat: float, lon: float, start: date, end: date) -> Json:
    if config.skip_weather:
        return {}
    cache_path = config.cache_dir / "open_meteo" / f"{lat:.5f}_{lon:.5f}_{start}_{end}.json"

    def fetch() -> Any:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(WEATHER_DAILY_VARS),
            "timezone": "UTC",
        }
        return request_json(session, OPEN_METEO_ARCHIVE_URL, params=params, timeout=config.request_timeout_s)

    data = cached_json(cache_path, fetch)
    return data if isinstance(data, dict) else {}


def aggregate_weather(weather_json: Mapping[str, Any]) -> Json:
    daily = weather_json.get("daily") if isinstance(weather_json, Mapping) else None
    if not isinstance(daily, Mapping):
        return {}

    def values(name: str) -> list[float]:
        raw = daily.get(name, [])
        if not isinstance(raw, list):
            return []
        clean = [_coerce_numeric(value) for value in raw]
        return [value for value in clean if value is not None]

    tmean = values("temperature_2m_mean")
    tmin = values("temperature_2m_min")
    tmax = values("temperature_2m_max")
    precip = values("precipitation_sum")
    rain = values("rain_sum")
    radiation = values("shortwave_radiation_sum")
    et0 = values("et0_fao_evapotranspiration")

    def mean(items: Sequence[float]) -> float | None:
        return sum(items) / len(items) if items else None

    def std(items: Sequence[float]) -> float | None:
        if len(items) < 2:
            return 0.0 if len(items) == 1 else None
        avg = mean(items) or 0.0
        return math.sqrt(sum((item - avg) ** 2 for item in items) / len(items))

    return {
        "season_weather_days": len(tmean),
        "season_weather_tmean_mean": mean(tmean),
        "season_weather_tmean_min": min(tmean) if tmean else None,
        "season_weather_tmean_max": max(tmean) if tmean else None,
        "season_weather_tmean_std": std(tmean),
        "season_weather_tmin_min": min(tmin) if tmin else None,
        "season_weather_tmax_max": max(tmax) if tmax else None,
        "season_weather_precip_sum": sum(precip) if precip else None,
        "season_weather_precip_mean": mean(precip),
        "season_weather_rain_sum": sum(rain) if rain else None,
        "season_weather_radiation_sum": sum(radiation) if radiation else None,
        "season_weather_radiation_mean": mean(radiation),
        "season_weather_et0_sum": sum(et0) if et0 else None,
        "season_weather_et0_mean": mean(et0),
        "season_weather_gdd_base5_sum": sum(max(0.0, item - GDD_BASE_C) for item in tmean) if tmean else None,
        "season_weather_heat_days": sum(1 for item in tmax if item >= HEAT_THRESHOLD_C),
        "season_weather_dry_days": sum(1 for item in precip if item < DRY_PRECIP_MM),
        "season_weather_frost_days": sum(1 for item in tmin if item < 0.0),
    }


def fetch_soilgrids(session: requests.Session, config: Config, lat: float, lon: float) -> Json:
    if config.skip_soil:
        return {}
    cache_path = config.cache_dir / "soilgrids" / f"{lat:.5f}_{lon:.5f}.json"

    def fetch() -> Any:
        params = {
            "lat": lat,
            "lon": lon,
            "property": list(SOIL_PROPERTIES),
            "depth": list(SOIL_DEPTHS),
            "value": "mean",
        }
        return request_json(session, SOILGRIDS_QUERY_URL, params=params, timeout=config.request_timeout_s)

    data = cached_json(cache_path, fetch)
    return data if isinstance(data, dict) else {}


def aggregate_soil(soil_json: Mapping[str, Any]) -> Json:
    features: Json = {}
    layers = soil_json.get("properties", {}).get("layers", []) if isinstance(soil_json, Mapping) else []
    if not isinstance(layers, list):
        return features
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        prop = str(layer.get("name", "")).strip()
        depths = layer.get("depths", [])
        depth_values: list[float] = []
        if not prop or not isinstance(depths, list):
            continue
        for depth_item in depths:
            if not isinstance(depth_item, Mapping):
                continue
            label = str(depth_item.get("label", "")).replace("-", "_").replace("cm", "cm")
            value = _coerce_numeric(depth_item.get("values", {}).get("mean") if isinstance(depth_item.get("values"), Mapping) else None)
            if value is None:
                continue
            features[f"soil_{prop}_{label}_mean"] = value
            depth_values.append(value)
        if depth_values:
            features[f"soil_{prop}_depth_mean"] = sum(depth_values) / len(depth_values)
    return features


def enrich_row(
    row: pd.Series,
    session: requests.Session,
    config: Config,
    field_cache: dict[str, Json],
    usage: SourceUsage,
) -> Json:
    output = {key: value for key, value in row.to_dict().items() if key not in LOCAL_SERIES_COLUMNS}
    output.update({"enrich_error": None, "cropwise_error": None, "weather_error": None, "soil_error": None, "coord_source": None})
    try:
        sowing = parse_date(row.get("sowing_date"))
        harvest = parse_date(row.get("harvesting_date"))
        if sowing and harvest:
            output["season_duration_days"] = (harvest - sowing).days
        else:
            output["season_duration_days"] = None
        output.update(aggregate_local_timeseries(row, sowing, harvest))
        usage.local_ts_rows += 1

        lat, lon = coords_from_row(row)
        area = area_from_row(row)
        if lat is not None and lon is not None:
            output["coord_source"] = "csv"

        field_id = str(row.get("field_id", "")).strip()
        if (lat is None or lon is None or area is None) and field_id and not config.skip_cropwise:
            try:
                field_json = fetch_cropwise_field(session, config, field_cache, field_id)
                if field_json:
                    usage.cropwise_rows += 1
                geometry = extract_field_geometry(field_json)
                cw_lat, cw_lon = compute_centroid(geometry)
                lat = lat if lat is not None else cw_lat
                lon = lon if lon is not None else cw_lon
                area = area if area is not None else field_area_from_cropwise(field_json)
                area = area if area is not None else _area_from_geometry(geometry)
                if lat is not None and lon is not None and output["coord_source"] is None:
                    output["coord_source"] = "cropwise"
            except CropwiseAuthError:
                raise
            except (requests.RequestException, ValueError, OSError) as exc:
                output["cropwise_error"] = exc.__class__.__name__
                logging.warning("Cropwise field %s failed: %s", field_id, exc.__class__.__name__)

        output["centroid_lat"] = lat
        output["centroid_lon"] = lon
        output["field_area"] = area

        if lat is None or lon is None:
            if not config.skip_weather:
                output["weather_error"] = "no_coordinates"
            if not config.skip_soil:
                output["soil_error"] = "no_coordinates"
            return output
        if sowing is None or harvest is None:
            output["weather_error"] = "no_dates"
        elif not config.skip_weather:
            try:
                output.update(aggregate_weather(fetch_weather(session, config, float(lat), float(lon), sowing, harvest)))
                usage.weather_rows += 1
            except (requests.RequestException, ValueError, OSError) as exc:
                output["weather_error"] = exc.__class__.__name__
                logging.warning("Weather lookup failed: %s", exc.__class__.__name__)
        if not config.skip_soil:
            try:
                output.update(aggregate_soil(fetch_soilgrids(session, config, float(lat), float(lon))))
                usage.soil_rows += 1
            except (requests.RequestException, ValueError, OSError) as exc:
                output["soil_error"] = exc.__class__.__name__
                logging.warning("SoilGrids lookup failed: %s", exc.__class__.__name__)
        return output
    except Exception as exc:
        output["enrich_error"] = exc.__class__.__name__
        usage.error_rows += 1
        if config.fail_fast:
            raise
        logging.error("Row failed: %s", exc.__class__.__name__)
        return output


def build_dataset(config: Config) -> tuple[pd.DataFrame, SourceUsage]:
    source = load_source_csv(config.input_path, config.max_rows)
    session = make_session()
    field_cache: dict[str, Json] = {}
    usage = SourceUsage()
    rows: list[Json] = []
    for processed_count, (_, row) in enumerate(source.iterrows(), start=1):
        try:
            rows.append(enrich_row(row, session, config, field_cache, usage))
        except CropwiseAuthError as exc:
            logging.error("%s; stopping Cropwise calls to avoid hammering API", exc)
            raise
        if processed_count % 25 == 0:
            logging.info("Processed %s rows", processed_count)
    return pd.DataFrame(rows), usage


def diagnose_cropwise(config: Config, session: requests.Session) -> int:
    print(f"Cropwise base URL: {config.cropwise_base_url}")
    print(f"Auth header mode: {config.cropwise_auth_header}")
    print(f"Token env source: {config.cropwise_token_source or 'not found'}")
    print(f"Token length: {len(config.cropwise_token or '')}")
    if not config.cropwise_token:
        print("ERROR: Cropwise token is not set")
        return 2
    try:
        fields = request_json(
            session,
            f"{config.cropwise_base_url.rstrip('/')}/fields",
            headers=cropwise_headers(config),
            params={"limit": 1},
            timeout=config.request_timeout_s,
        )
        print("/fields: OK")
        print(f"/fields response type: {type(fields).__name__}")
    except CropwiseAuthError as exc:
        print(f"/fields: {exc}")
        print("Try --cropwise-auth-header bearer, or verify the exact user API token/env.")
        return 2
    except requests.RequestException as exc:
        print(f"/fields: {exc.__class__.__name__}")
        return 1

    source = load_source_csv(config.input_path, 1)
    field_id = str(source.iloc[0].get("field_id", "")).strip()
    if not field_id:
        print("/fields/{id}: skipped; first row has no field_id")
        return 0
    try:
        field = fetch_cropwise_field(session, config, {}, field_id)
        geometry = extract_field_geometry(field)
        lat, lon = compute_centroid(geometry)
        print(f"/fields/{field_id}: OK")
        print(f"geometry: {'yes' if geometry else 'no'}; centroid: {lat}, {lon}")
        return 0
    except CropwiseAuthError as exc:
        print(f"/fields/{field_id}: {exc}")
        return 2
    except requests.RequestException as exc:
        print(f"/fields/{field_id}: {exc.__class__.__name__}")
        return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich yield CSV for forecasting.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", default=Path(".cache/cropwise_enrich"), type=Path)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-cropwise", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-soil", action="store_true")
    parser.add_argument("--cropwise-auth-header", choices=VALID_AUTH_HEADERS, default="x-user-api-token")
    parser.add_argument("--diagnose-cropwise", action="store_true")
    parser.add_argument("--output-format", choices=("csv", "parquet"), default=None)
    parser.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=True)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--cropwise-base-url", default=CROPWISE_BASE_URL)
    parser.add_argument(
        "--cropwise-token-env",
        default=None,
        help="Explicit token env name. If omitted, first of CROPWISE_USER_API_TOKEN, USER_API_TOKEN, X_USER_API_TOKEN, CROPWISE_TOKEN wins.",
    )
    parser.add_argument("--sleep-s", default=0.15, type=float)
    parser.add_argument("--request-timeout-s", default=60, type=int)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def parse_args() -> Config:
    load_dotenv_if_present()
    args = _parse_args()
    token, token_source = resolve_cropwise_token(args.cropwise_token_env)
    output_format = args.output_format or ("parquet" if args.output.suffix.lower() == ".parquet" else "csv")
    return Config(
        input_path=args.input,
        output_path=args.output,
        cache_dir=ensure_dir(args.cache_dir),
        output_format=output_format,
        cropwise_base_url=args.cropwise_base_url,
        cropwise_token=token,
        cropwise_token_source=token_source,
        cropwise_auth_header=args.cropwise_auth_header,
        max_rows=args.max_rows,
        sleep_s=args.sleep_s,
        request_timeout_s=args.request_timeout_s,
        skip_cropwise=args.skip_cropwise,
        skip_weather=args.skip_weather,
        skip_soil=args.skip_soil,
        fail_fast=args.fail_fast,
        diagnose_cropwise=args.diagnose_cropwise,
    )


def _write_output(df: pd.DataFrame, config: Config) -> None:
    ensure_dir(config.output_path.parent if str(config.output_path.parent) else Path("."))
    if config.output_format == "parquet":
        df.to_parquet(config.output_path, index=False)
    else:
        df.to_csv(config.output_path, index=False)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    load_dotenv_if_present()
    token, token_source = resolve_cropwise_token(args.cropwise_token_env)
    output_format = args.output_format or ("parquet" if args.output.suffix.lower() == ".parquet" else "csv")
    config = Config(
        input_path=args.input,
        output_path=args.output,
        cache_dir=ensure_dir(args.cache_dir),
        output_format=output_format,
        cropwise_base_url=args.cropwise_base_url,
        cropwise_token=token,
        cropwise_token_source=token_source,
        cropwise_auth_header=args.cropwise_auth_header,
        max_rows=args.max_rows,
        sleep_s=args.sleep_s,
        request_timeout_s=args.request_timeout_s,
        skip_cropwise=args.skip_cropwise,
        skip_weather=args.skip_weather,
        skip_soil=args.skip_soil,
        fail_fast=args.fail_fast,
        diagnose_cropwise=args.diagnose_cropwise,
    )
    if config.diagnose_cropwise:
        raise SystemExit(diagnose_cropwise(config, make_session()))
    if not config.skip_cropwise and not config.cropwise_token:
        logging.warning("Cropwise token env not found; Cropwise geometry lookup will be skipped")
        config = Config(**{**config.__dict__, "skip_cropwise": True})
    try:
        enriched, usage = build_dataset(config)
    except CropwiseAuthError:
        raise SystemExit(2)
    _write_output(enriched, config)
    added_columns = [column for column in enriched.columns if column not in load_source_csv(config.input_path, config.max_rows).columns]
    print(f"Saved: {config.output_path}")
    print(f"Rows: {len(enriched)}; columns: {len(enriched.columns)}; added_columns: {len(added_columns)}")
    print(f"Successful rows: {len(enriched) - usage.error_rows}; error rows: {usage.error_rows}")
    print(
        "External sources used: "
        f"cropwise_rows={usage.cropwise_rows}, weather_rows={usage.weather_rows}, "
        f"soil_rows={usage.soil_rows}, local_ts_rows={usage.local_ts_rows}"
    )
    print("Added columns:")
    print(", ".join(added_columns))
    print("First 5 rows:")
    print(enriched.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
