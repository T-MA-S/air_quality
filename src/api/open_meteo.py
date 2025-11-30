"""Open-Meteo API client for weather data."""

import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from src.config import settings
from src.utils.cache import file_cache
from src.utils.logger import setup_logger
from src.utils.rate_limiter import RateLimiter, RetryHandler

logger = setup_logger(__name__)


class OpenMeteoClient:
    """Client for Open-Meteo API."""

    def __init__(self, base_url: str = None, rate_limit: int = 100):
        """
        Initialize Open-Meteo client.

        Args:
            base_url: API base URL
            rate_limit: Requests per minute limit
        """
        self.base_url = base_url or settings.open_meteo_base_url
        self.rate_limiter = RateLimiter(max_requests=rate_limit, time_window=60)
        self.retry_handler = RetryHandler(max_retries=3)

        # Use more granular timeout settings
        timeout = httpx.Timeout(10.0, connect=10.0, read=60.0)
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and retries."""
        self.rate_limiter.wait_if_needed()

        cache_key = hashlib.md5(
            f"{endpoint}:{str(params)}".encode()
        ).hexdigest()

        # Check cache
        cached_response = file_cache.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_response

        def _request():
            response = self.client.get(endpoint, params=params or {})
            response.raise_for_status()
            return response.json()

        result = self.retry_handler.execute(_request)

        # Cache response
        file_cache.set(cache_key, result)

        return result

    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        timezone: str = "auto",
        hourly: List[str] = None,
        daily: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Get historical weather data.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            start_date: Start date
            end_date: End date
            timezone: Timezone (e.g., 'Europe/Moscow', 'UTC', 'auto')
            hourly: List of hourly parameters (e.g., ['temperature_2m', 'precipitation'])
            daily: List of daily parameters

        Returns:
            Weather data dictionary
        """
        if hourly is None:
            hourly = [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
            ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": timezone,
            "hourly": ",".join(hourly),
        }

        if daily:
            params["daily"] = ",".join(daily)

        return self._make_request("/forecast", params=params)

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        timezone: str = "auto",
        hourly: List[str] = None,
        daily: List[str] = None,
        forecast_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get weather forecast.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            timezone: Timezone
            hourly: List of hourly parameters
            daily: List of daily parameters
            forecast_days: Number of days to forecast

        Returns:
            Forecast data dictionary
        """
        if hourly is None:
            hourly = [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
            ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "hourly": ",".join(hourly),
            "forecast_days": forecast_days,
        }

        if daily:
            params["daily"] = ",".join(daily)

        return self._make_request("/forecast", params=params)

    def get_air_quality_forecast(
        self,
        latitude: float,
        longitude: float,
        timezone: str = "auto",
        forecast_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get air quality forecast (PM2.5, PM10, etc.).

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            timezone: Timezone
            forecast_days: Number of days to forecast

        Returns:
            Air quality forecast data
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide",
            "forecast_days": forecast_days,
        }

        return self._make_request("/air-quality", params=params)

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

