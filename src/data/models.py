"""Data models for cities and coordinates."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class City:
    """City metadata."""

    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    openaq_city_name: Optional[str] = None  # May differ from name

    def __post_init__(self):
        """Validate city data."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")


# Predefined cities with coordinates
CITIES = [
    City(
        name="London",
        country="GB",
        latitude=51.5074,
        longitude=-0.1278,
        timezone="Europe/London",
        openaq_city_name="London",
    ),
    City(
        name="Paris",
        country="FR",
        latitude=48.8566,
        longitude=2.3522,
        timezone="Europe/Paris",
        openaq_city_name="Paris",
    ),
    City(
        name="Berlin",
        country="DE",
        latitude=52.5200,
        longitude=13.4050,
        timezone="Europe/Berlin",
        openaq_city_name="Berlin",
    ),
    City(
        name="New York",
        country="US",
        latitude=40.7128,
        longitude=-74.0060,
        timezone="America/New_York",
        openaq_city_name="New York",
    ),
    City(
        name="Tokyo",
        country="JP",
        latitude=35.6762,
        longitude=139.6503,
        timezone="Asia/Tokyo",
        openaq_city_name="Tokyo",
    ),
]


def get_city_by_name(name: str) -> Optional[City]:
    """Get city by name."""
    for city in CITIES:
        if city.name.lower() == name.lower():
            return city
    return None

