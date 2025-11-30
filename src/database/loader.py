"""Data loading into database."""

from typing import List, Optional
import pandas as pd
from sqlalchemy import insert, select, and_

from src.database.connection import get_session, get_engine
from src.database.schema import dim_city, dim_metric, fact_air_quality, fact_weather
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load data into database."""

    def __init__(self):
        """Initialize data loader."""
        self.engine = get_engine()

    def load_cities(self, cities: List[dict]) -> dict:
        """
        Load cities into dim_city table.

        Returns:
            Dictionary mapping city names to city_ids
        """
        city_id_map = {}
        session = get_session()

        try:
            for city_data in cities:
                # Check if city exists
                stmt = select(dim_city).where(
                    dim_city.c.city_name == city_data["name"]
                )
                result = session.execute(stmt).first()

                if result:
                    city_id_map[city_data["name"]] = result.city_id
                else:
                    # Insert new city
                    stmt = insert(dim_city).values(
                        city_name=city_data["name"],
                        country=city_data["country"],
                        latitude=city_data["latitude"],
                        longitude=city_data["longitude"],
                        timezone=city_data["timezone"],
                        openaq_city_name=city_data.get("openaq_city_name"),
                    ).returning(dim_city.c.city_id)
                    result = session.execute(stmt)
                    city_id = result.scalar()
                    city_id_map[city_data["name"]] = city_id
                    logger.info(f"Inserted city: {city_data['name']} (ID: {city_id})")

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading cities: {e}")
            raise
        finally:
            session.close()

        return city_id_map

    def get_city_id(self, city_name: str) -> Optional[int]:
        """Get city_id by city name."""
        session = get_session()
        try:
            stmt = select(dim_city.c.city_id).where(
                dim_city.c.city_name == city_name
            )
            result = session.execute(stmt).scalar()
            return result
        finally:
            session.close()

    def get_metric_id(self, metric_code: str) -> Optional[int]:
        """Get metric_id by metric code."""
        session = get_session()
        try:
            stmt = select(dim_metric.c.metric_id).where(
                dim_metric.c.metric_code == metric_code
            )
            result = session.execute(stmt).scalar()
            return result
        finally:
            session.close()

    def load_air_quality(self, df: pd.DataFrame, city_id_map: dict) -> int:
        """
        Load air quality data into fact_air_quality table.

        Returns:
            Number of records inserted
        """
        if df.empty:
            return 0

        session = get_session()
        records_inserted = 0

        try:
            # Get metric IDs
            metric_id_map = {}
            for metric_code in df.columns:
                if metric_code in ["pm25", "pm10", "no2", "o3", "co", "so2"]:
                    metric_id = self.get_metric_id(metric_code)
                    if metric_id:
                        metric_id_map[metric_code] = metric_id

            # Prepare records and check for duplicates
            records = []
            for _, row in df.iterrows():
                city_id = city_id_map.get(row["city"])
                if not city_id:
                    continue

                timestamp = pd.to_datetime(row["timestamp"])

                # Insert one record per metric
                for metric_code, metric_id in metric_id_map.items():
                    if metric_code in row and pd.notna(row[metric_code]):
                        # Check if record already exists (avoid duplicates)
                        check_stmt = select(fact_air_quality).where(
                            and_(
                                fact_air_quality.c.timestamp == timestamp,
                                fact_air_quality.c.city_id == city_id,
                                fact_air_quality.c.metric_id == metric_id,
                            )
                        )
                        existing = session.execute(check_stmt).first()
                        
                        if not existing:
                            record = {
                                "timestamp": timestamp,
                                "city_id": city_id,
                                "metric_id": metric_id,
                                "value": float(row[metric_code]),
                                "value_ug_m3": float(row[metric_code]),
                                "station_count": int(row.get("station_count", 1)),
                                "latitude": float(row.get("latitude", 0)),
                                "longitude": float(row.get("longitude", 0)),
                                "location_id": str(row.get("location_id", ""))[:50],
                                "location_name": str(row.get("location_name", ""))[:200],
                            }
                            records.append(record)

            # Bulk insert
            if records:
                stmt = insert(fact_air_quality)
                session.execute(stmt, records)
                session.commit()
                records_inserted = len(records)
                logger.info(f"Inserted {records_inserted} new air quality records (skipped duplicates)")
            else:
                logger.info("No new records to insert (all duplicates)")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading air quality data: {e}")
            raise
        finally:
            session.close()

        return records_inserted

    def load_weather(self, df: pd.DataFrame, city_id_map: dict) -> int:
        """
        Load weather data into fact_weather table.

        Returns:
            Number of records inserted
        """
        if df.empty:
            return 0

        session = get_session()
        records_inserted = 0

        try:
            records = []
            for _, row in df.iterrows():
                city_id = city_id_map.get(row["city"])
                if not city_id:
                    continue

                timestamp = pd.to_datetime(row["timestamp"])

                record = {
                    "timestamp": timestamp,
                    "city_id": city_id,
                    "temperature_c": float(row.get("temperature_c", 0)) if pd.notna(row.get("temperature_c")) else None,
                    "humidity_percent": float(row.get("humidity_percent", 0)) if pd.notna(row.get("humidity_percent")) else None,
                    "precipitation_mm": float(row.get("precipitation_mm", 0)) if pd.notna(row.get("precipitation_mm")) else None,
                    "wind_speed_ms": float(row.get("wind_speed_ms", 0)) if pd.notna(row.get("wind_speed_ms")) else None,
                    "wind_direction_deg": float(row.get("wind_direction_deg", 0)) if pd.notna(row.get("wind_direction_deg")) else None,
                    "latitude": float(row.get("latitude", 0)),
                    "longitude": float(row.get("longitude", 0)),
                }
                records.append(record)

            # Bulk insert
            if records:
                stmt = insert(fact_weather)
                session.execute(stmt, records)
                session.commit()
                records_inserted = len(records)
                logger.info(f"Inserted {records_inserted} weather records")

        except Exception as e:
            session.rollback()
            logger.error(f"Error loading weather data: {e}")
            raise
        finally:
            session.close()

        return records_inserted

