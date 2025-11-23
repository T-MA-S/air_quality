"""Integration tests."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.etl.pipeline import ETLPipeline
from src.data.models import City


class TestETLPipeline:
    """Integration tests for ETL pipeline."""

    @patch("src.api.openaq.OpenAQClient")
    @patch("src.api.open_meteo.OpenMeteoClient")
    @patch("src.database.loader.DataLoader")
    def test_pipeline_initialization(self, mock_loader, mock_meteo, mock_openaq):
        """Test pipeline initialization."""
        cities = [City("Test", "US", 40.7128, -74.0060, "America/New_York")]
        pipeline = ETLPipeline(cities=cities)

        assert len(pipeline.cities) == 1
        assert pipeline.openaq_client is not None
        assert pipeline.meteo_client is not None

    @patch("src.database.schema.create_engine")
    @patch("src.api.openaq.OpenAQClient")
    @patch("src.api.open_meteo.OpenMeteoClient")
    def test_extract_air_quality(self, mock_meteo, mock_openaq, mock_engine):
        """Test air quality extraction."""
        mock_client = Mock()
        mock_client.get_measurements.return_value = [
            {
                "location": {"id": 1, "name": "S1", "coordinates": {"latitude": 40.7, "longitude": -74.0}},
                "parameter": "pm25",
                "value": 15.0,
                "unit": "µg/m³",
                "date": {"utc": "2024-01-01T12:00:00Z"},
            }
        ]
        mock_openaq.return_value = mock_client

        city = City("Test", "US", 40.7128, -74.0060, "America/New_York")
        pipeline = ETLPipeline(cities=[city])

        date_from = datetime.now() - timedelta(days=1)
        date_to = datetime.now()

        df = pipeline.extract_air_quality(city, date_from, date_to, parameters=["pm25"])

        assert not df.empty
        mock_client.get_measurements.assert_called()

