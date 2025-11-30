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

        # Get exactly 5 locations from 25km radius
        all_locations = []
        try:
            locations = self.openaq_client.get_locations(
                latitude=city.latitude,
                longitude=city.longitude,
                radius=25000,  # 25km
                limit=5  # Take only 5 locations
            )
            all_locations = locations[:5]  # Ensure we have max 5
            if all_locations:
                logger.info(f"Found {len(all_locations)} locations near {city.name} (radius: 25km, limit: 5)")
            else:
                logger.warning(f"No locations found for {city.name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting locations for {city.name}: {e}")
            return pd.DataFrame()

        # Analyze parameters in these 5 locations
        location_param_map = {}  # {location_id: set of parameters}
        for loc in all_locations:
            loc_id = loc.get("id")
            if not loc_id:
                continue
            sensors = loc.get("sensors", [])
            loc_params = set()
            for sensor in sensors:
                param_info = sensor.get("parameter", {})
                if isinstance(param_info, dict):
                    param_name = param_info.get("name", "").lower()
                    if param_name in parameters:
                        loc_params.add(param_name)
            if loc_params:
                location_param_map[loc_id] = loc_params
                logger.debug(f"Location {loc_id} has parameters: {sorted(loc_params)}")

        # Check if we have all parameters in these 5 locations
        all_found_params = set()
        for loc_params in location_param_map.values():
            all_found_params |= loc_params
        
        missing_params = set(parameters) - all_found_params
        
        if missing_params:
            logger.warning(f"Missing parameters in 5 locations for {city.name}: {sorted(missing_params)}")
            # Try to find missing parameters in additional locations
            logger.info(f"Searching for missing parameters in additional locations...")
            try:
                additional_locations = self.openaq_client.get_locations(
                    latitude=city.latitude,
                    longitude=city.longitude,
                    radius=25000,
                    limit=20  # Get more to find missing params
                )
                # Skip already checked locations
                checked_ids = set(location_param_map.keys())
                for loc in additional_locations:
                    loc_id = loc.get("id")
                    if loc_id in checked_ids:
                        continue
                    sensors = loc.get("sensors", [])
                    loc_params = set()
                    for sensor in sensors:
                        param_info = sensor.get("parameter", {})
                        if isinstance(param_info, dict):
                            param_name = param_info.get("name", "").lower()
                            if param_name in missing_params:
                                loc_params.add(param_name)
                    if loc_params:
                        location_param_map[loc_id] = loc_params
                        missing_params -= loc_params
                        logger.info(f"Found missing parameter(s) {sorted(loc_params)} in location {loc_id}")
                        if not missing_params:
                            break
            except Exception as e:
                logger.debug(f"Error searching for missing parameters: {e}")

        # Select locations to cover all required parameters
        selected_location_ids = []
        found_params = set()
        
        # First, try to find a location with all parameters
        for loc_id, loc_params in location_param_map.items():
            if loc_params >= set(parameters):
                selected_location_ids = [loc_id]
                found_params = loc_params
                logger.info(f"Found location {loc_id} with all parameters for {city.name}")
                break
        
        # If no single location has all parameters, select multiple locations
        if not selected_location_ids:
            remaining_params = set(parameters)
            # Sort by number of needed parameters (prefer locations with more missing params)
            sorted_locations = sorted(
                location_param_map.items(),
                key=lambda x: len(x[1] & remaining_params),
                reverse=True
            )
            
            for loc_id, loc_params in sorted_locations:
                if remaining_params & loc_params:  # If location has any missing params
                    selected_location_ids.append(loc_id)
                    found_params |= (loc_params & remaining_params)
                    new_params = loc_params & remaining_params
                    remaining_params -= loc_params
                    logger.info(f"Selected location {loc_id} for {city.name} (adds: {sorted(new_params)})")
                    if not remaining_params:
                        break
            
            if remaining_params:
                logger.warning(f"Could not find locations with parameters {sorted(remaining_params)} for {city.name}")
            else:
                logger.info(f"Selected {len(selected_location_ids)} locations to cover all parameters for {city.name}")

        if not selected_location_ids:
            logger.warning(f"No locations found with required parameters for {city.name}")
            return pd.DataFrame()

        # Get measurements from selected locations
        all_measurements = []
        for loc_id in selected_location_ids:
            try:
                # Get all latest measurements from this location
                measurements = self.openaq_client.get_measurements(
                    location_id=loc_id,
                    latitude=None,
                    longitude=None,
                    parameter=None,  # Get all parameters from this location
                    date_from=date_from,
                    date_to=date_to,
                )
                
                # Filter by requested parameters
                for meas in measurements:
                    param_name = meas.get("parameter", "").lower()
                    if param_name in parameters:
                        all_measurements.append(meas)
                
                logger.info(
                    f"Extracted {len(measurements)} measurements from location {loc_id} for {city.name}"
                )
                    
            except Exception as e:
                logger.debug(f"Error getting data from location {loc_id} for {city.name}: {e}")
                continue
        
        # Log what parameters we found
        found_params = {meas.get("parameter", "").lower() for meas in all_measurements if meas.get("parameter", "").lower() in parameters}
        missing_params = set(parameters) - found_params
        if missing_params:
            logger.warning(f"Missing parameters for {city.name}: {missing_params}")
        if found_params:
            logger.info(f"Found parameters for {city.name}: {found_params}")

        if not all_measurements:
            logger.warning(f"No air quality data found for {city.name}")
            return pd.DataFrame()

        # Transform measurements
        df = transform_openaq_measurements(all_measurements, city)
        
        # Update timestamps to current hour (since /latest returns data with old timestamps)
        # This ensures we have data marked with current collection time
        if not df.empty:
            # Round to current hour
            current_hour = pd.Timestamp.now(tz=df["timestamp"].iloc[0].tz).floor("H")
            df["timestamp"] = current_hour
        
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
                aq_records = 0
                weather_records = 0
                
                if not aq_df.empty:
                    logger.info(f"Loading {len(aq_df)} air quality records for {city.name} into database")
                    aq_records = self.data_loader.load_air_quality(
                        aq_df, city_id_map
                    )
                    results["total_aq_records"] += aq_records
                    logger.info(f"Successfully loaded {aq_records} air quality records for {city.name}")
                else:
                    logger.warning(f"No air quality data to load for {city.name}")

                if not weather_df.empty:
                    logger.info(f"Loading {len(weather_df)} weather records for {city.name} into database")
                    weather_records = self.data_loader.load_weather(
                        weather_df, city_id_map
                    )
                    results["total_weather_records"] += weather_records
                    logger.info(f"Successfully loaded {weather_records} weather records for {city.name}")
                else:
                    logger.warning(f"No weather data to load for {city.name}")

                results["cities_processed"] += 1
                logger.info(f"Successfully processed {city.name}: {aq_records} AQ records, {weather_records} weather records")

            except Exception as e:
                logger.error(f"Error processing {city.name}: {e}", exc_info=True)
                results["cities_failed"] += 1

        # Close clients
        self.openaq_client.close()
        self.meteo_client.close()

        logger.info(f"ETL pipeline completed: {results}")
        return results

