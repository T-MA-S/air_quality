"""OpenAQ API client for air quality data."""

import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from src.config import settings
from src.utils.cache import file_cache
from src.utils.logger import setup_logger
from src.utils.rate_limiter import RateLimiter, RetryHandler

logger = setup_logger(__name__)


class OpenAQClient:
    """Client for OpenAQ API."""

    def __init__(
        self,
        base_url: str = None,
        api_key: Optional[str] = None,
        rate_limit: int = None,
    ):
        """
        Initialize OpenAQ client.

        Args:
            base_url: API base URL
            api_key: Optional API key
            rate_limit: Requests per minute limit
        """
        self.base_url = base_url or settings.openaq_api_base_url
        self.api_key = api_key or settings.openaq_api_key
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit or settings.openaq_rate_limit, time_window=60
        )
        self.retry_handler = RetryHandler(max_retries=3)

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
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

    def get_locations(
        self,
        city: Optional[str] = None,
        country: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get locations (stations) for a city or country.

        Args:
            city: City name filter
            country: Country code filter (ISO 3166-1 alpha-2)
            limit: Maximum number of results

        Returns:
            List of location dictionaries
        """
        params = {"limit": limit}
        if city:
            params["city"] = city
        if country:
            params["country"] = country

        response = self._make_request("/locations", params=params)
        return response.get("results", [])

    def get_measurements(
        self,
        location_id: Optional[int] = None,
        city: Optional[str] = None,
        parameter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Get air quality measurements.

        Args:
            location_id: Specific location ID
            city: City name filter
            parameter: Parameter code (e.g., 'pm25', 'pm10', 'no2', 'o3')
            date_from: Start date
            date_to: End date
            limit: Maximum number of results

        Returns:
            List of measurement dictionaries
        """
        params = {"limit": limit}

        if location_id:
            params["location_id"] = location_id
        if city:
            params["city"] = city
        if parameter:
            params["parameter"] = parameter
        if date_from:
            params["date_from"] = date_from.isoformat()
        if date_to:
            params["date_to"] = date_to.isoformat()

        response = self._make_request("/measurements", params=params)
        return response.get("results", [])

    def get_cities(self, country: Optional[str] = None, limit: int = 1000) -> List[str]:
        """
        Get list of cities with available data.

        Args:
            country: Country code filter
            limit: Maximum number of results

        Returns:
            List of city names
        """
        params = {"limit": limit}
        if country:
            params["country"] = country

        response = self._make_request("/cities", params=params)
        cities = [city.get("city") for city in response.get("results", [])]
        return [c for c in cities if c]  # Filter out None values

    def get_latest_measurements(
        self,
        city: Optional[str] = None,
        parameter: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get latest measurements for locations.

        Args:
            city: City name filter
            parameter: Parameter code filter
            limit: Maximum number of results

        Returns:
            List of latest measurement dictionaries
        """
        params = {"limit": limit}
        if city:
            params["city"] = city
        if parameter:
            params["parameter"] = parameter

        response = self._make_request("/latest", params=params)
        return response.get("results", [])

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

