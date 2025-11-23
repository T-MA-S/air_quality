"""Main ETL pipeline."""

from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from src.api.openaq import OpenAQClient
from src.api.open_meteo import OpenMeteoClient
from src.data.models import City, CITIES
from src.data.transform import (
    transform_openaq_measurements,
    transform_open_meteo_weather,
    merge_air_quality_and_weather,
)
from src.data.quality import DataQualityChecker
from src.database.loader import DataLoader
from src.database.schema import create_schema
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ETLPipeline:
    """Main ETL pipeline for air quality and weather data."""

    def __init__(self, cities: List[City] = None):
        """
        Initialize ETL pipeline.

        Args:
            cities: List of cities to process (default: all predefined cities)
        """
        self.cities = cities or CITIES
        self.openaq_client = OpenAQClient()
        self.meteo_client = OpenMeteoClient()
        self.data_loader = DataLoader()
        self.quality_checker = DataQualityChecker()

    def extract_air_quality(
        self,
        city: City,
        date_from: datetime,
        date_to: datetime,
        parameters: List[str] = None,
    ) -> pd.DataFrame:
        """
        Extract air quality data for a city.

        Args:
            city: City metadata
            date_from: Start date
            date_to: End date
            parameters: List of parameters to extract (default: pm25, pm10, no2, o3)

        Returns:
            DataFrame with air quality data
        """
        if parameters is None:
            parameters = ["pm25", "pm10", "no2", "o3"]

        logger.info(
            f"Extracting air quality data for {city.name} from {date_from} to {date_to}"
        )

        all_measurements = []
        for param in parameters:
            try:
                # Use coordinates instead of city name for v3 API
                measurements = self.openaq_client.get_measurements(
                    latitude=city.latitude,
                    longitude=city.longitude,
                    parameter=param,
                    date_from=date_from,
                    date_to=date_to,
                )
                all_measurements.extend(measurements)
                logger.info(
                    f"Extracted {len(measurements)} measurements for {param} in {city.name}"
                )
            except Exception as e:
                logger.error(f"Error extracting {param} for {city.name}: {e}")

        if not all_measurements:
            logger.warning(f"No air quality data found for {city.name}")
            return pd.DataFrame()

        df = transform_openaq_measurements(all_measurements, city)
        return df

    def extract_weather(
        self, city: City, date_from: datetime, date_to: datetime
    ) -> pd.DataFrame:
        """
        Extract weather data for a city.

        Args:
            city: City metadata
            date_from: Start date
            date_to: End date

        Returns:
            DataFrame with weather data
        """
        logger.info(
            f"Extracting weather data for {city.name} from {date_from} to {date_to}"
        )

        try:
            weather_data = self.meteo_client.get_historical_weather(
                latitude=city.latitude,
                longitude=city.longitude,
                start_date=date_from,
                end_date=date_to,
                timezone=city.timezone,
            )
            df = transform_open_meteo_weather(weather_data, city)
            logger.info(f"Extracted {len(df)} weather records for {city.name}")
            return df
        except Exception as e:
            logger.error(f"Error extracting weather for {city.name}: {e}")
            return pd.DataFrame()

    def run(
        self,
        date_from: datetime,
        date_to: datetime,
        cities: Optional[List[City]] = None,
    ) -> dict:
        """
        Run complete ETL pipeline.

        Args:
            date_from: Start date
            date_to: End date
            cities: Optional list of cities (default: use self.cities)

        Returns:
            Dictionary with execution results
        """
        cities_to_process = cities or self.cities
        results = {
            "cities_processed": 0,
            "cities_failed": 0,
            "total_aq_records": 0,
            "total_weather_records": 0,
            "quality_issues": [],
        }

        # Initialize database schema
        try:
            create_schema()
        except Exception as e:
            logger.warning(f"Schema might already exist: {e}")

        # Load cities
        city_data = [
            {
                "name": city.name,
                "country": city.country,
                "latitude": city.latitude,
                "longitude": city.longitude,
                "timezone": city.timezone,
                "openaq_city_name": city.openaq_city_name,
            }
            for city in cities_to_process
        ]
        city_id_map = self.data_loader.load_cities(city_data)

        # Process each city
        for city in cities_to_process:
            try:
                logger.info(f"Processing city: {city.name}")

                # Extract
                aq_df = self.extract_air_quality(city, date_from, date_to)
                weather_df = self.extract_weather(city, date_from, date_to)

                if aq_df.empty and weather_df.empty:
                    logger.warning(f"No data extracted for {city.name}")
                    results["cities_failed"] += 1
                    continue

                # Quality checks
                if not aq_df.empty:
                    aq_report = self.quality_checker.validate_air_quality_data(aq_df)
                    if not aq_report["valid"]:
                        results["quality_issues"].append(
                            {"city": city.name, "type": "air_quality", "issues": aq_report["warnings"]}
                        )

                # Transform and merge
                if not aq_df.empty and not weather_df.empty:
                    merged_df = merge_air_quality_and_weather(aq_df, weather_df)
                elif not aq_df.empty:
                    merged_df = aq_df
                elif not weather_df.empty:
                    merged_df = weather_df
                else:
                    merged_df = pd.DataFrame()

                # Load
                if not aq_df.empty:
                    aq_records = self.data_loader.load_air_quality(
                        aq_df, city_id_map
                    )
                    results["total_aq_records"] += aq_records

                if not weather_df.empty:
                    weather_records = self.data_loader.load_weather(
                        weather_df, city_id_map
                    )
                    results["total_weather_records"] += weather_records

                results["cities_processed"] += 1
                logger.info(f"Successfully processed {city.name}")

            except Exception as e:
                logger.error(f"Error processing {city.name}: {e}", exc_info=True)
                results["cities_failed"] += 1

        # Close clients
        self.openaq_client.close()
        self.meteo_client.close()

        logger.info(f"ETL pipeline completed: {results}")
        return results

