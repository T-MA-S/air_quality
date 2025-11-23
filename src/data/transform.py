"""Data transformation and normalization."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from src.data.models import City
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def normalize_timestamp(dt: Any, timezone_str: str = "UTC") -> pd.Timestamp:
    """
    Normalize timestamp to pandas Timestamp with timezone.

    Args:
        dt: Datetime object or string
        timezone_str: Target timezone

    Returns:
        Normalized pandas Timestamp
    """
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)

    if isinstance(dt, datetime):
        dt = pd.Timestamp(dt)

    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")

    return dt.tz_convert(timezone_str)


def transform_openaq_measurements(
    measurements: List[Dict[str, Any]], city: City
) -> pd.DataFrame:
    """
    Transform OpenAQ measurements to standardized format.
    Supports both v2 and v3 API formats.

    Args:
        measurements: List of measurement dictionaries from OpenAQ
        city: City metadata

    Returns:
        DataFrame with standardized columns
    """
    if not measurements:
        return pd.DataFrame()

    records = []
    for m in measurements:
        try:
            # OpenAQ v3 structure differs from v2
            # Try v3 format first, then fall back to v2
            
            # Extract location info (v3 may have different structure)
            location = m.get("location", {})
            if not location:
                # v3 might have location data directly in measurement
                location = m
            
            # Extract coordinates
            coordinates = location.get("coordinates", {})
            if not coordinates:
                # v3 might have coordinates directly
                if "latitude" in location and "longitude" in location:
                    coordinates = {
                        "latitude": location.get("latitude"),
                        "longitude": location.get("longitude")
                    }
                elif "geometry" in location:
                    # v3 might use GeoJSON geometry
                    geom = location.get("geometry", {})
                    if geom.get("type") == "Point":
                        coords = geom.get("coordinates", [])
                        if len(coords) >= 2:
                            coordinates = {
                                "longitude": coords[0],
                                "latitude": coords[1]
                            }

            # Normalize timestamp
            # v3 uses "datetime" or "date" field
            date_utc = None
            if "datetime" in m:
                date_utc = m.get("datetime")
            elif "date" in m:
                date_obj = m.get("date")
                if isinstance(date_obj, dict):
                    date_utc = date_obj.get("utc")
                else:
                    date_utc = date_obj
            elif "dateUTC" in m:
                date_utc = m.get("dateUTC")
            
            if date_utc:
                timestamp = normalize_timestamp(date_utc, city.timezone)
            else:
                continue

            # Extract parameter (v3 might use parameter object or string)
            parameter = m.get("parameter", "")
            if isinstance(parameter, dict):
                parameter = parameter.get("name", parameter.get("id", ""))
            parameter = str(parameter).lower()

            # Extract value
            value = m.get("value", 0)
            if isinstance(value, dict):
                value = value.get("value", 0)

            # Extract unit
            unit = m.get("unit", "")
            if isinstance(unit, dict):
                unit = unit.get("name", unit.get("symbol", ""))
            unit = str(unit).lower()

            record = {
                "timestamp": timestamp,
                "city": city.name,
                "country": city.country,
                "parameter": parameter,
                "value": float(value),
                "unit": unit,
                "latitude": coordinates.get("latitude") or city.latitude,
                "longitude": coordinates.get("longitude") or city.longitude,
                "location_id": location.get("id") or location.get("locationId"),
                "location_name": location.get("name") or location.get("locationName"),
            }

            # Convert units to standard (micrograms per cubic meter)
            if record["unit"] in ["µg/m³", "ug/m3", "ug/m^3", "micrograms per cubic meter"]:
                record["value_ug_m3"] = record["value"]
            elif record["unit"] == "ppm":
                # Approximate conversion (varies by parameter)
                record["value_ug_m3"] = record["value"] * 1000  # Rough approximation
            else:
                record["value_ug_m3"] = record["value"]

            records.append(record)
        except Exception as e:
            logger.warning(f"Error transforming measurement: {e}, skipping")
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Aggregate by city and hour (if multiple stations)
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")

    return df


def transform_open_meteo_weather(
    weather_data: Dict[str, Any], city: City
) -> pd.DataFrame:
    """
    Transform Open-Meteo weather data to standardized format.

    Args:
        weather_data: Weather data dictionary from Open-Meteo
        city: City metadata

    Returns:
        DataFrame with standardized columns
    """
    if "hourly" not in weather_data:
        return pd.DataFrame()

    hourly = weather_data["hourly"]
    time_data = hourly.get("time", [])

    if not time_data:
        return pd.DataFrame()

    # Create base DataFrame with timestamps
    df = pd.DataFrame({"timestamp": pd.to_datetime(time_data)})
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(city.timezone)
    df["city"] = city.name
    df["country"] = city.country
    df["latitude"] = city.latitude
    df["longitude"] = city.longitude

    # Extract hourly parameters
    param_mapping = {
        "temperature_2m": "temperature_c",
        "relative_humidity_2m": "humidity_percent",
        "precipitation": "precipitation_mm",
        "wind_speed_10m": "wind_speed_ms",
        "wind_direction_10m": "wind_direction_deg",
    }

    for api_param, df_column in param_mapping.items():
        if api_param in hourly:
            df[df_column] = hourly[api_param]

    # Create hourly floor for aggregation
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")

    return df


def merge_air_quality_and_weather(
    air_quality_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge air quality and weather data by timestamp and city.

    Args:
        air_quality_df: Air quality DataFrame
        weather_df: Weather DataFrame

    Returns:
        Merged DataFrame
    """
    if air_quality_df.empty or weather_df.empty:
        return pd.DataFrame()

    # Aggregate air quality by hour and parameter
    aq_agg = (
        air_quality_df.groupby(["city", "timestamp_hour", "parameter"])
        .agg(
            {
                "value_ug_m3": ["mean", "count"],
                "latitude": "first",
                "longitude": "first",
            }
        )
        .reset_index()
    )
    aq_agg.columns = [
        "city",
        "timestamp_hour",
        "parameter",
        "value_ug_m3_mean",
        "station_count",
        "latitude",
        "longitude",
    ]

    # Pivot parameters to columns
    aq_pivot = aq_agg.pivot_table(
        index=["city", "timestamp_hour", "latitude", "longitude", "station_count"],
        columns="parameter",
        values="value_ug_m3_mean",
        aggfunc="mean",
    ).reset_index()

    # Merge with weather
    weather_agg = (
        weather_df.groupby(["city", "timestamp_hour"])
        .agg(
            {
                "temperature_c": "mean",
                "humidity_percent": "mean",
                "precipitation_mm": "sum",
                "wind_speed_ms": "mean",
                "wind_direction_deg": "mean",
            }
        )
        .reset_index()
    )

    merged = pd.merge(
        aq_pivot,
        weather_agg,
        on=["city", "timestamp_hour"],
        how="outer",
    )

    merged["timestamp"] = merged["timestamp_hour"]
    merged = merged.drop(columns=["timestamp_hour"])

    return merged

