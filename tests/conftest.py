"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.data.models import City
from src.data.quality import DataQualityChecker


@pytest.fixture
def sample_city():
    """Sample city for testing."""
    return City(
        name="Test City",
        country="US",
        latitude=40.7128,
        longitude=-74.0060,
        timezone="America/New_York",
    )


@pytest.fixture
def sample_air_quality_data():
    """Sample air quality DataFrame."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq="H",
    )
    return pd.DataFrame(
        {
            "timestamp": dates,
            "city": ["Test City"] * len(dates),
            "parameter": ["pm25"] * len(dates),
            "value_ug_m3": [10 + i % 20 for i in range(len(dates))],
            "latitude": [40.7128] * len(dates),
            "longitude": [-74.0060] * len(dates),
        }
    )


@pytest.fixture
def sample_weather_data():
    """Sample weather DataFrame."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq="H",
    )
    return pd.DataFrame(
        {
            "timestamp": dates,
            "city": ["Test City"] * len(dates),
            "temperature_c": [20 + i % 10 for i in range(len(dates))],
            "humidity_percent": [50 + i % 20 for i in range(len(dates))],
            "precipitation_mm": [0] * len(dates),
            "wind_speed_ms": [5 + i % 5 for i in range(len(dates))],
            "wind_direction_deg": [180 + i % 180 for i in range(len(dates))],
        }
    )


@pytest.fixture
def quality_checker():
    """Data quality checker instance."""
    return DataQualityChecker(max_missing_ratio=0.3, min_data_points_per_day=20)

