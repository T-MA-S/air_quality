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
        else:
            # OpenAQ v3 requires API key - warn if not provided
            logger.warning("OpenAQ API key not provided. v3 API requires authentication.")

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
            params["citiesName"] = city  # v3 uses citiesName instead of city
        if country:
            params["countriesId"] = country  # v3 may use countriesId

        response = self._make_request("/locations", params=params)
        # v3 returns data directly in response, not in "results"
        if "results" in response:
            return response.get("results", [])
        elif "data" in response:
            return response.get("data", [])
        else:
            return response if isinstance(response, list) else []

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
        Get air quality measurements using OpenAQ v3 API.
        
        Note: OpenAQ v3 API uses different structure. We use /latest endpoint
        which returns the most recent measurements. For historical data,
        v3 API may require different endpoints or approach.

        Args:
            location_id: Specific location ID
            city: City name filter
            parameter: Parameter code (e.g., 'pm25', 'pm10', 'no2', 'o3')
            date_from: Start date (may not be fully supported in v3 latest endpoint)
            date_to: End date (may not be fully supported in v3 latest endpoint)
            limit: Maximum number of results

        Returns:
            List of measurement dictionaries
        """
        params = {"limit": limit}
        
        # v3 latest endpoint supports these filters
        if location_id:
            params["locationsId"] = location_id
        if city:
            params["citiesName"] = city
        if parameter:
            # v3 uses parametersName for parameter codes
            params["parametersName"] = parameter
        
        # Note: date_from and date_to may not be supported in /latest endpoint
        # For historical data, v3 might require different approach
        
        try:
            response = self._make_request("/latest", params=params)
            
            # v3 returns data in "results" field
            if "results" in response:
                measurements = response.get("results", [])
            elif isinstance(response, list):
                measurements = response
            else:
                measurements = []
            
            # Transform to v2-like format for compatibility
            transformed = []
            for m in measurements:
                # v3 structure: location contains measurements
                location = m.get("location", {})
                measurements_list = m.get("measurements", [])
                
                for meas in measurements_list:
                    # Extract parameter info
                    param_info = meas.get("parameter", {})
                    param_name = param_info.get("name", "") if isinstance(param_info, dict) else str(param_info)
                    
                    # Create v2-compatible structure
                    transformed.append({
                        "location": location,
                        "parameter": param_name,
                        "value": meas.get("value"),
                        "unit": meas.get("unit") or param_info.get("units", "") if isinstance(param_info, dict) else "",
                        "date": {
                            "utc": meas.get("date", {}).get("utc") if isinstance(meas.get("date"), dict) else meas.get("date")
                        }
                    })
            
            return transformed[:limit]
            
        except Exception as e:
            logger.error(f"Error getting measurements from v3 API: {e}")
            return []

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
            params["countriesId"] = country  # v3 may use countriesId

        response = self._make_request("/cities", params=params)
        # v3 structure may differ
        if "results" in response:
            cities = [city.get("name") or city.get("city") for city in response.get("results", [])]
        elif "data" in response:
            cities = [city.get("name") or city.get("city") for city in response.get("data", [])]
        else:
            cities = []
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
            params["citiesName"] = city  # v3 uses citiesName
        if parameter:
            params["parametersName"] = parameter  # v3 uses parametersName for parameter codes

        response = self._make_request("/latest", params=params)
        # v3 returns data directly in response, not in "results"
        if "results" in response:
            return response.get("results", [])
        elif "data" in response:
            return response.get("data", [])
        else:
            return response if isinstance(response, list) else []

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

