"""Database schema definitions and initialization."""

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Index,
    Text,
    MetaData,
    Table,
    create_engine,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP

from src.config import settings
from src.database.connection import get_engine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

metadata = MetaData()

# Dimension tables
dim_city = Table(
    "dim_city",
    metadata,
    Column("city_id", Integer, primary_key=True, autoincrement=True),
    Column("city_name", String(100), nullable=False),
    Column("country", String(2), nullable=False),
    Column("latitude", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("timezone", String(50), nullable=False),
    Column("openaq_city_name", String(100)),
    Column("created_at", TIMESTAMP(timezone=True), server_default="now()"),
    Column("updated_at", TIMESTAMP(timezone=True), server_default="now()"),
    Index("idx_dim_city_name", "city_name"),
)

dim_metric = Table(
    "dim_metric",
    metadata,
    Column("metric_id", Integer, primary_key=True, autoincrement=True),
    Column("metric_code", String(20), nullable=False, unique=True),
    Column("metric_name", String(100), nullable=False),
    Column("unit", String(20), nullable=False),
    Column("description", Text),
    Column("created_at", TIMESTAMP(timezone=True), server_default="now()"),
    Index("idx_dim_metric_code", "metric_code"),
)

# Fact tables
fact_air_quality = Table(
    "fact_air_quality",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", TIMESTAMP(timezone=True), nullable=False),
    Column("city_id", Integer, nullable=False),
    Column("metric_id", Integer, nullable=False),
    Column("value", Float, nullable=False),
    Column("value_ug_m3", Float),
    Column("station_count", Integer),
    Column("latitude", Float),
    Column("longitude", Float),
    Column("location_id", String(50)),
    Column("location_name", String(200)),
    Column("created_at", TIMESTAMP(timezone=True), server_default="now()"),
    Index("idx_fact_aq_timestamp", "timestamp"),
    Index("idx_fact_aq_city_metric", "city_id", "metric_id"),
    Index("idx_fact_aq_city_timestamp", "city_id", "timestamp"),
)

fact_weather = Table(
    "fact_weather",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", TIMESTAMP(timezone=True), nullable=False),
    Column("city_id", Integer, nullable=False),
    Column("temperature_c", Float),
    Column("humidity_percent", Float),
    Column("precipitation_mm", Float),
    Column("wind_speed_ms", Float),
    Column("wind_direction_deg", Float),
    Column("latitude", Float),
    Column("longitude", Float),
    Column("created_at", TIMESTAMP(timezone=True), server_default="now()"),
    Index("idx_fact_weather_timestamp", "timestamp"),
    Index("idx_fact_weather_city_timestamp", "city_id", "timestamp"),
)


def create_schema():
    """Create all tables in the database."""
    engine = get_engine()
    metadata.create_all(engine)
    logger.info("Database schema created")

    # Insert default metrics
    _insert_default_metrics(engine)


def _insert_default_metrics(engine):
    """Insert default metric definitions."""
    from sqlalchemy import insert

    default_metrics = [
        {"metric_code": "pm25", "metric_name": "PM2.5", "unit": "µg/m³", "description": "Particulate matter 2.5 micrometers"},
        {"metric_code": "pm10", "metric_name": "PM10", "unit": "µg/m³", "description": "Particulate matter 10 micrometers"},
        {"metric_code": "no2", "metric_name": "Nitrogen Dioxide", "unit": "µg/m³", "description": "NO₂ concentration"},
        {"metric_code": "o3", "metric_name": "Ozone", "unit": "µg/m³", "description": "O₃ concentration"},
        {"metric_code": "co", "metric_name": "Carbon Monoxide", "unit": "µg/m³", "description": "CO concentration"},
        {"metric_code": "so2", "metric_name": "Sulfur Dioxide", "unit": "µg/m³", "description": "SO₂ concentration"},
    ]

    with engine.begin() as conn:
        for metric in default_metrics:
            try:
                stmt = insert(dim_metric).values(**metric)
                conn.execute(stmt)
            except Exception as e:
                # Metric might already exist
                logger.debug(f"Metric {metric['metric_code']} might already exist: {e}")
                pass  # begin() handles rollback automatically

