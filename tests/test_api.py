"""Tests for API clients."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.api.openaq import OpenAQClient
from src.api.open_meteo import OpenMeteoClient


class TestOpenAQClient:
    """Tests for OpenAQ client."""

    @patch("src.api.openaq.httpx.Client")
    def test_client_initialization(self, mock_client_class):
        """Test client initialization."""
        client = OpenAQClient()
        assert client.base_url == "https://api.openaq.org/v2"
        assert client.rate_limiter is not None

    @patch("src.api.openaq.httpx.Client")
    def test_get_locations(self, mock_client_class):
        """Test getting locations."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"id": 1, "name": "Station 1", "city": "Moscow"},
                {"id": 2, "name": "Station 2", "city": "Moscow"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = OpenAQClient()
        locations = client.get_locations(city="Moscow")

        assert len(locations) == 2
        assert locations[0]["city"] == "Moscow"

    @patch("src.api.openaq.httpx.Client")
    def test_rate_limiting(self, mock_client_class):
        """Test rate limiting."""
        client = OpenAQClient(rate_limit=10)
        assert client.rate_limiter.max_requests == 10


class TestOpenMeteoClient:
    """Tests for Open-Meteo client."""

    @patch("src.api.open_meteo.httpx.Client")
    def test_client_initialization(self, mock_client_class):
        """Test client initialization."""
        client = OpenMeteoClient()
        assert "open-meteo.com" in client.base_url
        assert client.rate_limiter is not None

    @patch("src.api.open_meteo.httpx.Client")
    def test_get_historical_weather(self, mock_client_class):
        """Test getting historical weather."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [10.0, 11.0],
                "precipitation": [0.0, 0.0],
            }
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = OpenMeteoClient()
        date_from = datetime.now() - timedelta(days=7)
        date_to = datetime.now()

        weather = client.get_historical_weather(
            latitude=55.7558,
            longitude=37.6173,
            start_date=date_from,
            end_date=date_to,
        )

        assert "hourly" in weather
        assert len(weather["hourly"]["time"]) == 2

