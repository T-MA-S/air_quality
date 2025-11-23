"""Tests for data transformation."""

import pytest
import pandas as pd
from datetime import datetime

from src.data.transform import (
    normalize_timestamp,
    transform_openaq_measurements,
    transform_open_meteo_weather,
    merge_air_quality_and_weather,
)
from src.data.models import City


class TestTransform:
    """Tests for transformation functions."""

    def test_normalize_timestamp(self):
        """Test timestamp normalization."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        ts = normalize_timestamp(dt, "Europe/Moscow")
        assert ts.tz is not None
        assert ts.tz.zone == "Europe/Moscow"

    def test_transform_openaq_measurements(self, sample_city):
        """Test OpenAQ measurements transformation."""
        measurements = [
            {
                "location": {
                    "id": 1,
                    "name": "Station 1",
                    "coordinates": {"latitude": 40.7128, "longitude": -74.0060},
                },
                "parameter": "pm25",
                "value": 15.5,
                "unit": "µg/m³",
                "date": {"utc": "2024-01-01T12:00:00Z"},
            }
        ]

        df = transform_openaq_measurements(measurements, sample_city)

        assert not df.empty
        assert "timestamp" in df.columns
        assert "parameter" in df.columns
        assert "value_ug_m3" in df.columns
        assert df["parameter"].iloc[0] == "pm25"

    def test_transform_open_meteo_weather(self, sample_city):
        """Test Open-Meteo weather transformation."""
        weather_data = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [10.0, 11.0],
                "relative_humidity_2m": [50.0, 55.0],
                "precipitation": [0.0, 0.5],
                "wind_speed_10m": [5.0, 6.0],
                "wind_direction_10m": [180.0, 185.0],
            }
        }

        df = transform_open_meteo_weather(weather_data, sample_city)

        assert not df.empty
        assert "temperature_c" in df.columns
        assert "humidity_percent" in df.columns
        assert len(df) == 2

    def test_merge_air_quality_and_weather(self, sample_air_quality_data, sample_weather_data):
        """Test merging air quality and weather data."""
        merged = merge_air_quality_and_weather(sample_air_quality_data, sample_weather_data)

        assert not merged.empty
        assert "pm25" in merged.columns or "value_ug_m3_mean" in merged.columns
        assert "temperature_c" in merged.columns

