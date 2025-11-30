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

        # Use more granular timeout settings
        # connect timeout: time to establish connection
        # read timeout: time to read response (OpenAQ can be very slow with large responses)
        timeout = httpx.Timeout(10.0, connect=10.0, read=180.0)
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
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
            try:
                logger.debug(f"Making request to {endpoint} with params: {params}")
                response = self.client.get(endpoint, params=params or {})
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException as e:
                logger.error(f"Timeout error for {endpoint}: {e}")
                raise
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error for {endpoint}: {e.response.status_code} - {e.response.text[:200]}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                raise

        result = self.retry_handler.execute(_request)

        # Cache response
        file_cache.set(cache_key, result)

        return result

    def get_locations(
        self,
        city: Optional[str] = None,
        country: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius: int = 5000,  # radius in meters (5km default, sufficient for city center)
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get locations (stations) for a city, country, or coordinates.

        Args:
            city: City name filter (may not work in v3)
            country: Country code filter (ISO 3166-1 alpha-2)
            latitude: Latitude for coordinate-based search
            longitude: Longitude for coordinate-based search
            radius: Search radius in meters (default 5km, max 25km in v3)
            limit: Maximum number of results

        Returns:
            List of location dictionaries
        """
        params = {"limit": limit}
        
        # v3 API: use coordinates if provided (more reliable)
        if latitude is not None and longitude is not None:
            # v3 uses coordinates parameter: "latitude,longitude" (not lon,lat!)
            # And radius must be <= 25000 (25km max)
            params["coordinates"] = f"{latitude},{longitude}"
            params["radius"] = min(radius, 25000)  # v3 API max radius is 25km
        elif city:
            # Try city name (may not work for all cities)
            params["citiesName"] = city
        elif country:
            # Try country code
            params["countriesId"] = country

        response = self._make_request("/locations", params=params)
        # v3 returns data in "results" field
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
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        parameter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Get air quality measurements using OpenAQ v3 API.
        
        Strategy: First find locations near the city coordinates,
        then get latest measurements for those locations.

        Args:
            location_id: Specific location ID
            city: City name filter
            latitude: Latitude for coordinate-based search
            longitude: Longitude for coordinate-based search
            parameter: Parameter code (e.g., 'pm25', 'pm10', 'no2', 'o3')
            date_from: Start date (not fully supported - latest endpoint returns recent data only)
            date_to: End date (not fully supported - latest endpoint returns recent data only)
            limit: Maximum number of results

        Returns:
            List of measurement dictionaries
        """
        all_measurements = []
        
        # First, get locations
        location_ids = []
        if location_id:
            location_ids = [location_id]
        elif latitude is not None and longitude is not None:
            # Find locations near coordinates (use smaller radius and limit for faster response)
            locations = self.get_locations(latitude=latitude, longitude=longitude, radius=5000, limit=10)
            location_ids = [loc.get("id") for loc in locations if loc.get("id")]
            logger.info(f"Found {len(location_ids)} locations near coordinates ({latitude}, {longitude})")
        elif city:
            # Try to find locations by city name
            locations = self.get_locations(city=city, limit=50)
            location_ids = [loc.get("id") for loc in locations if loc.get("id")]
            logger.info(f"Found {len(location_ids)} locations for city {city}")
        
        if not location_ids:
            logger.warning(f"No locations found for city={city}, lat={latitude}, lon={longitude}")
            return []
        
        # For each location, get latest measurements
        # NOTE: OpenAQ v3 /latest endpoint returns only the most recent measurements
        # It does NOT support historical date ranges. For historical data,
        # we would need to use a different approach (e.g., collect data regularly over time)
        # For now, we get latest available data which may be recent or old depending on station activity
        for loc_id in location_ids:
            try:
                # Get latest measurements for this location
                # This returns the most recent available data (may be hours or days old)
                endpoint = f"/locations/{loc_id}/latest"
                response = self._make_request(endpoint, params={})
                
                # v3 returns measurements in "results" field
                measurements = response.get("results", [])
                
                # Also get location info for context
                location_endpoint = f"/locations/{loc_id}"
                location_data = self._make_request(location_endpoint, params={})
                location_info = location_data.get("results", [{}])[0] if location_data.get("results") else {}
                
                # Get sensor info to map sensorsId to parameter names
                sensors = location_info.get("sensors", [])
                sensor_param_map = {}
                for sensor in sensors:
                    sensor_id = sensor.get("id")
                    param_info = sensor.get("parameter", {})
                    if sensor_id and isinstance(param_info, dict):
                        sensor_param_map[sensor_id] = {
                            "name": param_info.get("name", "").lower(),
                            "units": param_info.get("units", "")
                        }
                
                # Transform v3 measurements to v2-like format
                for meas in measurements:
                    sensor_id = meas.get("sensorsId")
                    param_info = sensor_param_map.get(sensor_id, {})
                    param_name = param_info.get("name", "")
                    
                    # Filter by parameter if specified (if None, get all parameters)
                    if parameter is not None and param_name != parameter.lower():
                        continue
                    
                    # Skip if parameter name is empty
                    if not param_name:
                        continue
                    
                    # Extract datetime
                    dt_info = meas.get("datetime", {})
                    date_utc = dt_info.get("utc") if isinstance(dt_info, dict) else meas.get("datetime")
                    
                    # Create v2-compatible structure
                    all_measurements.append({
                        "location": {
                            "id": loc_id,
                            "name": location_info.get("name", ""),
                            "coordinates": meas.get("coordinates", location_info.get("coordinates", {}))
                        },
                        "parameter": param_name,
                        "value": meas.get("value", 0),
                        "unit": param_info.get("units", ""),
                        "date": {
                            "utc": date_utc
                        }
                    })
                
                if len(all_measurements) >= limit:
                    break
                    
            except Exception as e:
                logger.debug(f"Error getting data for location {loc_id}: {e}")
                continue
        
        logger.info(f"Retrieved {len(all_measurements)} measurements")
        return all_measurements[:limit]

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

